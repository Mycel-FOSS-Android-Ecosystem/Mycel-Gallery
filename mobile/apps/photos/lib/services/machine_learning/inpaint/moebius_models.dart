import "package:logging/logging.dart";
import "package:photos/services/remote_assets_service.dart";
import "package:photos/src/rust/api/inpaint_api.dart"
    show RustInpaintModelPaths;

/// Delivery of the Moebius inpainting ONNX graphs.
///
/// The three graphs are downloaded on first use (gated behind the AI-edit entry
/// point) and cached on device by [RemoteAssetsService]. The Rust pipeline loads
/// them from the returned filesystem paths; nothing is loaded in Dart.
///
/// NOTE: for now these are fp16 graphs hosted on a temporary dev asset host
/// (~0.6 GB total). Once validated they will move to the Ente CDN.
class MoebiusModels {
  static final _logger = Logger("MoebiusModels");

  static const String _base = "https://entedevassets.priem.dev/";

  // The U-Net is the einsum-rewritten variant (`unet_fp16_rw.onnx`): its 105
  // Einsum nodes are decomposed offline into Transpose/Reshape/MatMul, which
  // ONNX Runtime executes with fast multithreaded GEMM kernels instead of its
  // slow, mostly single-threaded Einsum kernel (~1.5x faster per denoise step
  // on a Pixel 8; numerically exact, max|delta| 1.8e-6 at fp32).
  static const String unetUrl = "${_base}unet_fp16_rw.onnx";
  static const String vaeEncoderUrl = "${_base}vae_encoder_fp16.onnx";
  static const String vaeDecoderUrl = "${_base}vae_decoder_fp16.onnx";

  // SHA-256 of the fp16 graphs.
  static const String _unetSha =
      "b2a99db24a297b3db251c529da7a526dc55097fac35455be6f6a72f32eb0d637";
  static const String _vaeEncoderSha =
      "c4a8c399498bea2c5817e1701f5a59e312a5be0a09139157f7934743eecb98e8";
  static const String _vaeDecoderSha =
      "2c51ab793f17a91246ce97c0f553751355b9de9b896051739d72762ab1c9c0fd";

  /// Approximate combined on-disk size, used for the first-run download prompt.
  static const int approxTotalBytes = 460093716 + 101260614 + 70585594;

  /// All three model URLs, useful for filtering [RemoteAssetsService.progressStream].
  static const List<String> urls = [unetUrl, vaeEncoderUrl, vaeDecoderUrl];

  /// True if all three graphs are already cached on device.
  static Future<bool> isDownloaded() async {
    final svc = RemoteAssetsService.instance;
    return await svc.hasAsset(unetUrl) &&
        await svc.hasAsset(vaeEncoderUrl) &&
        await svc.hasAsset(vaeDecoderUrl);
  }

  /// Ensures all three graphs are present (downloading if needed) and returns
  /// their on-device paths for the Rust pipeline.
  static Future<RustInpaintModelPaths> ensureDownloaded() async {
    final svc = RemoteAssetsService.instance;
    _logger.info("Resolving Moebius model paths (downloading if needed)");
    final unet = await svc.getAssetPath(unetUrl, expectedSha256: _unetSha);
    final vaeEncoder = await svc.getAssetPath(
      vaeEncoderUrl,
      expectedSha256: _vaeEncoderSha,
    );
    final vaeDecoder = await svc.getAssetPath(
      vaeDecoderUrl,
      expectedSha256: _vaeDecoderSha,
    );
    return RustInpaintModelPaths(
      unet: unet,
      vaeEncoder: vaeEncoder,
      vaeDecoder: vaeDecoder,
    );
  }
}
