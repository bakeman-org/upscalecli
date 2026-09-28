import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:path_provider/path_provider.dart';

/// Returns the directory where downloaded results are written.
///
/// On Android this is the app-private external dir:
///   /storage/emulated/0/Android/data/<pkg>/files/results
/// which is:
///   * writable without any runtime permission on every Android version
///   * visible to the user via Files / most file managers
///   * removed when the app is uninstalled
///
/// If [getExternalStorageDirectory] is unavailable (e.g. desktop, iOS), we
/// fall back to the app documents directory.
Future<Directory> resultStoreDir() async {
  Directory? base;
  try {
    base = await getExternalStorageDirectory();
  } catch (_) {}
  base ??= await getApplicationDocumentsDirectory();

  final dir = Directory('${base.path}${Platform.pathSeparator}results');
  if (!dir.existsSync()) dir.createSync(recursive: true);
  return dir;
}

/// Copy [sourcePath] into a location the user chooses via the Storage Access
/// Framework. On Android this opens the system "Save as" dialog, which grants
/// our app write access to exactly that one file — no permission required.
/// Returns the destination path/URI, or null if the user cancelled.
Future<String?> saveToPublicDownloads({
  required String sourcePath,
  required String suggestedName,
}) async {
  final bytes = await File(sourcePath).readAsBytes();
  final target = await FilePicker.platform.saveFile(
    dialogTitle: 'Save comic',
    fileName: suggestedName,
    bytes: bytes,
  );
  return target;
}
