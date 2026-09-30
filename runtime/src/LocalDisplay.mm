// SPDX-License-Identifier: MPL-2.0

#include "LocalDisplay.h"

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <atomic>
#include <mutex>
#include <spdlog/spdlog.h>

// This file is compiled without ARC, matching Swapchain.mm.

namespace
{

const char* kLocalDisplayMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct VertexOut
{
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut local_display_vertex(uint vid [[vertex_id]])
{
    // Full-viewport triangle; uv origin top-left like the swapchain textures.
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VertexOut out;
    out.position = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    out.uv = uv;
    return out;
}

struct EyeUniforms
{
    float4 tanFov;      // left, right, up, down (tangents)
    float4 warpColumn0; // display-eye -> render-eye rotation, column major
    float4 warpColumn1;
    float4 warpColumn2;
};

fragment float4 local_display_fragment(VertexOut in [[stage_in]],
                                       texture2d<float> eye [[texture(0)]],
                                       sampler linearSampler [[sampler(0)]],
                                       constant EyeUniforms& u [[buffer(0)]])
{
    // Ray through this pixel in the display-time eye frame (-Z forward, +Y up).
    float3 ray = float3(mix(u.tanFov.x, u.tanFov.y, in.uv.x),
                        mix(u.tanFov.z, u.tanFov.w, in.uv.y),
                        -1.0);
    float3x3 warp = float3x3(u.warpColumn0.xyz, u.warpColumn1.xyz, u.warpColumn2.xyz);
    float3 rendered = warp * ray;
    if (rendered.z >= -1e-4)
    {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    float2 tangent = rendered.xy / -rendered.z;
    float2 uv = float2((tangent.x - u.tanFov.x) / (u.tanFov.y - u.tanFov.x),
                       (u.tanFov.z - tangent.y) / (u.tanFov.z - u.tanFov.w));
    if (any(uv < 0.0) || any(uv > 1.0))
    {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    return float4(eye.sample(linearSampler, uv).rgb, 1.0);
}
)METAL";

struct EyeUniforms
{
    float tanFov[4];
    float warpColumns[3][4];
};

NSScreen* FindScreen(const std::string& name)
{
    NSString* wanted = [NSString stringWithUTF8String:name.c_str()];
    for (NSScreen* screen in [NSScreen screens])
    {
        if (wanted.length > 0 &&
            [screen.localizedName rangeOfString:wanted options:NSCaseInsensitiveSearch].location != NSNotFound)
        {
            return screen;
        }
    }
    return nil;
}

} // namespace

struct LocalDisplay::Impl
{
    Settings settings;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLSamplerState> sampler = nil;

    std::mutex pipelineMutex;
    MTLPixelFormat pipelineFormat = MTLPixelFormatInvalid;
    id<MTLRenderPipelineState> pipeline = nil;

    // Owned by the main thread; the render thread only reads the layer pointer
    // and target drawable size.
    NSWindow* window = nil;
    id screenObserver = nil;
    std::atomic<CAMetalLayer*> layer{nil};
    std::atomic<uint32_t> targetWidth{0};
    std::atomic<uint32_t> targetHeight{0};
    std::atomic<uint32_t> refreshHz{0};
    std::atomic_bool fallbackWindow{false};

    // Main thread only.
    void LayoutWindow()
    {
        NSScreen* screen = FindScreen(settings.screenName);
        const bool fallback = screen == nil;
        if (fallback)
        {
            screen = [NSScreen mainScreen];
        }

        NSRect frame = screen.frame;
        if (fallback)
        {
            // Preview: a 32:9 window so both eyes are visible side by side.
            frame = NSMakeRect(NSMinX(frame) + 40, NSMinY(frame) + 80, 1280, 360);
        }

        if (window == nil)
        {
            window = [[NSWindow alloc] initWithContentRect:frame
                                                 styleMask:(fallback ? NSWindowStyleMaskTitled : NSWindowStyleMaskBorderless)
                                                   backing:NSBackingStoreBuffered
                                                     defer:NO
                                                    screen:screen];
            window.releasedWhenClosed = NO;
            window.backgroundColor = [NSColor blackColor];
            window.collectionBehavior = NSWindowCollectionBehaviorCanJoinAllSpaces |
                                        NSWindowCollectionBehaviorFullScreenAuxiliary |
                                        NSWindowCollectionBehaviorStationary;

            CAMetalLayer* metalLayer = [CAMetalLayer layer];
            metalLayer.device = device;
            metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
            metalLayer.framebufferOnly = YES;
            metalLayer.displaySyncEnabled = YES;
            metalLayer.maximumDrawableCount = 3;
            NSView* view = window.contentView;
            view.layer = metalLayer; // layer-hosting view: set the layer before wantsLayer
            view.wantsLayer = YES;
            layer.store(metalLayer);
        }

        window.styleMask = fallback ? NSWindowStyleMaskTitled : NSWindowStyleMaskBorderless;
        window.title = @"OXRSys local display (no XREAL screen found)";
        window.level = fallback ? NSNormalWindowLevel : NSScreenSaverWindowLevel;
        window.ignoresMouseEvents = !fallback;
        [window setFrame:(fallback ? [window frameRectForContentRect:frame] : frame) display:YES];
        [window orderFrontRegardless];

        const CGFloat scale = screen.backingScaleFactor;
        const NSSize size = window.contentView.bounds.size;
        CAMetalLayer* metalLayer = layer.load();
        metalLayer.contentsScale = scale;
        targetWidth = static_cast<uint32_t>(size.width * scale);
        targetHeight = static_cast<uint32_t>(size.height * scale);
        refreshHz = static_cast<uint32_t>(std::max<NSInteger>(screen.maximumFramesPerSecond, 30));

        if (fallbackWindow.exchange(fallback) != fallback || !fallback)
        {
            spdlog::info("LocalDisplay: {} '{}' {}x{} @ {}Hz ({})",
                         fallback ? "no matching screen, preview on" : "presenting on",
                         screen.localizedName.UTF8String, targetWidth.load(), targetHeight.load(),
                         refreshHz.load(),
                         static_cast<float>(targetWidth) > 2.5f * static_cast<float>(targetHeight)
                             ? "side-by-side stereo" : "left eye only; enable 3D/SBS mode on the glasses for stereo");
        }
    }

    id<MTLRenderPipelineState> PipelineFor(MTLPixelFormat format)
    {
        std::scoped_lock lock(pipelineMutex);
        if (pipeline != nil && pipelineFormat == format)
        {
            return pipeline;
        }
        MTLRenderPipelineDescriptor* descriptor = [[MTLRenderPipelineDescriptor alloc] init];
        id<MTLFunction> vertexFunction = [library newFunctionWithName:@"local_display_vertex"];
        id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"local_display_fragment"];
        descriptor.vertexFunction = vertexFunction;
        descriptor.fragmentFunction = fragmentFunction;
        descriptor.colorAttachments[0].pixelFormat = format;
        NSError* error = nil;
        id<MTLRenderPipelineState> created = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
        [vertexFunction release];
        [fragmentFunction release];
        [descriptor release];
        if (created == nil)
        {
            spdlog::error("LocalDisplay: failed to create pipeline: {}",
                          error != nil ? error.localizedDescription.UTF8String : "unknown error");
            return nil;
        }
        [pipeline release];
        pipeline = created;
        pipelineFormat = format;
        return pipeline;
    }
};

LocalDisplay::LocalDisplay(void* metalDevice, const Settings& settings)
    : impl_(std::make_unique<Impl>())
{
    impl_->settings = settings;
    impl_->device = [(id<MTLDevice>)metalDevice retain];
    impl_->queue = [impl_->device newCommandQueue];
    impl_->queue.label = @"OXRSys LocalDisplay";

    NSError* error = nil;
    impl_->library = [impl_->device newLibraryWithSource:[NSString stringWithUTF8String:kLocalDisplayMetalSource]
                                                 options:nil
                                                   error:&error];
    if (impl_->library == nil)
    {
        spdlog::error("LocalDisplay: failed to compile shader: {}",
                      error != nil ? error.localizedDescription.UTF8String : "unknown error");
    }

    MTLSamplerDescriptor* samplerDescriptor = [[MTLSamplerDescriptor alloc] init];
    samplerDescriptor.minFilter = MTLSamplerMinMagFilterLinear;
    samplerDescriptor.magFilter = MTLSamplerMinMagFilterLinear;
    samplerDescriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
    samplerDescriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
    impl_->sampler = [impl_->device newSamplerStateWithDescriptor:samplerDescriptor];
    [samplerDescriptor release];

    // AppKit windows must be created on the main thread. The OpenXR frame loop may
    // run on a worker thread while the main thread waits on it, so never block here.
    Impl* impl = impl_.get();
    dispatch_async(dispatch_get_main_queue(), ^{
        impl->LayoutWindow();
        impl->screenObserver = [[[NSNotificationCenter defaultCenter]
            addObserverForName:NSApplicationDidChangeScreenParametersNotification
                        object:nil
                         queue:[NSOperationQueue mainQueue]
                    usingBlock:^(NSNotification*) { impl->LayoutWindow(); }] retain];
    });
}

LocalDisplay::~LocalDisplay()
{
    // Tear down window state on the main thread; the Impl moves into the block so
    // a still-pending creation block above runs first and sees a live Impl.
    Impl* impl = impl_.release();
    dispatch_async(dispatch_get_main_queue(), ^{
        if (impl->screenObserver != nil)
        {
            [[NSNotificationCenter defaultCenter] removeObserver:impl->screenObserver];
            [impl->screenObserver release];
        }
        [impl->window orderOut:nil];
        [impl->window close];
        [impl->window release];
        [impl->pipeline release];
        [impl->sampler release];
        [impl->library release];
        [impl->queue release];
        [impl->device release];
        delete impl;
    });
}

uint32_t LocalDisplay::GetRefreshRateHz() const
{
    return impl_->refreshHz.load();
}

void LocalDisplay::Present(FrameSource&& frame, const XrFovf fov[2],
                           const glm::quat& renderOrientation, const glm::quat& displayOrientation)
{
    CAMetalLayer* metalLayer = impl_->layer.load();
    if (metalLayer == nil || impl_->library == nil || !frame.IsStereoValid())
    {
        return;
    }

    @autoreleasepool
    {
        id<MTLTexture> left = (id<MTLTexture>)frame.left.GetImage();
        id<MTLTexture> right = (id<MTLTexture>)frame.right.GetImage();

        // Sampling an sRGB texture decodes it. For linear content an sRGB drawable
        // encodes it back; for already gamma-encoded content (see Settings) a
        // plain drawable keeps the decode, undoing the swapchain's encoding.
        const bool srgb = left.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB ||
                          left.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB;
        const MTLPixelFormat format = srgb && !impl_->settings.gammaEncodedSources
            ? MTLPixelFormatBGRA8Unorm_sRGB : MTLPixelFormatBGRA8Unorm;
        if (metalLayer.pixelFormat != format)
        {
            spdlog::info("LocalDisplay: eye texture format {} ({}x{}, type {}), drawable format {}",
                         static_cast<int>(left.pixelFormat), left.width, left.height,
                         static_cast<int>(left.textureType), static_cast<int>(format));
            metalLayer.pixelFormat = format;
        }
        const CGSize size = CGSizeMake(impl_->targetWidth.load(), impl_->targetHeight.load());
        if (size.width < 1 || size.height < 1)
        {
            return;
        }
        if (!CGSizeEqualToSize(metalLayer.drawableSize, size))
        {
            metalLayer.drawableSize = size;
        }

        id<MTLRenderPipelineState> pipeline = impl_->PipelineFor(format);
        id<CAMetalDrawable> drawable = [metalLayer nextDrawable];
        id<MTLCommandBuffer> commandBuffer = [impl_->queue commandBuffer];
        if (pipeline == nil || drawable == nil || commandBuffer == nil)
        {
            return;
        }

        // GPU-side wait for the app's release-time snapshot of each eye.
        for (const FrameImageSource* source : {&frame.left, &frame.right})
        {
            if (source->sync.api == GraphicsApi::Metal && source->sync.IsValid())
            {
                [commandBuffer encodeWaitForEvent:(id<MTLSharedEvent>)source->sync.waitObject.get()
                                            value:source->sync.waitValue];
            }
        }

        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = drawable.texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:pass];
        [encoder setRenderPipelineState:pipeline];
        [encoder setFragmentSamplerState:impl_->sampler atIndex:0];

        // Rotation taking a display-time eye ray into the render-time eye frame.
        const glm::mat3 warp = impl_->settings.timewarp
            ? glm::mat3_cast(glm::inverse(renderOrientation) * displayOrientation)
            : glm::mat3(1.0f);

        const bool sideBySide = size.width > 2.5 * size.height;
        const uint32_t eyeCount = sideBySide ? 2 : 1;
        const double eyeWidth = size.width / eyeCount;
        for (uint32_t eyeIndex = 0; eyeIndex < eyeCount; ++eyeIndex)
        {
            EyeUniforms uniforms = {};
            uniforms.tanFov[0] = std::tan(fov[eyeIndex].angleLeft);
            uniforms.tanFov[1] = std::tan(fov[eyeIndex].angleRight);
            uniforms.tanFov[2] = std::tan(fov[eyeIndex].angleUp);
            uniforms.tanFov[3] = std::tan(fov[eyeIndex].angleDown);
            for (int column = 0; column < 3; ++column)
            {
                for (int row = 0; row < 3; ++row)
                {
                    uniforms.warpColumns[column][row] = warp[column][row];
                }
            }

            MTLViewport viewport = {eyeWidth * eyeIndex, 0.0, eyeWidth, size.height, 0.0, 1.0};
            [encoder setViewport:viewport];
            [encoder setFragmentTexture:(eyeIndex == 0 ? left : right) atIndex:0];
            [encoder setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:0];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
        [encoder endEncoding];

        // Keep the eye images (and their staging leases) alive until the GPU is done.
        auto keepAlive = std::make_shared<FrameSource>(std::move(frame));
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
            keepAlive->Reset();
        }];
        [commandBuffer presentDrawable:drawable];
        [commandBuffer commit];
    }
}
