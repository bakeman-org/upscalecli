import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:dio/dio.dart';

import '../models/task.dart';

class UploadResult {
  final String taskId;
  final String status;
  UploadResult(this.taskId, this.status);
}

class UpscaleApi {
  final String baseUrl;
  final Dio _dio;

  UpscaleApi({required this.baseUrl})
      : _dio = Dio(BaseOptions(
          baseUrl: baseUrl,
          connectTimeout: const Duration(seconds: 15),
          receiveTimeout: const Duration(seconds: 30),
        ));

  Future<void> health() async {
    await _dio.get('/api/health');
  }

  Future<UploadResult> upload({
    required String filePath,
    required String fileName,
    String? net,
    int? scale,
    void Function(int sent, int total)? onProgress,
    CancelToken? cancelToken,
  }) async {
    final form = FormData.fromMap({
      'file': await MultipartFile.fromFile(filePath, filename: fileName),
      if (net != null && net.isNotEmpty) 'net': net,
      if (scale != null) 'scale': scale.toString(),
    });

    final resp = await _dio.post(
      '/api/tasks',
      data: form,
      onSendProgress: onProgress,
      cancelToken: cancelToken,
      options: Options(
        sendTimeout: const Duration(hours: 2),
        receiveTimeout: const Duration(seconds: 60),
      ),
    );

    final data = resp.data as Map<String, dynamic>;
    return UploadResult(data['task_id'] as String, data['status'] as String);
  }

  Future<Task> getTask(String id) async {
    final resp = await _dio.get('/api/tasks/$id');
    return Task.fromJson(resp.data as Map<String, dynamic>);
  }

  Future<List<Task>> listTasks() async {
    final resp = await _dio.get('/api/tasks');
    final list = (resp.data['tasks'] as List).cast<Map<String, dynamic>>();
    return list.map(Task.fromJson).toList();
  }

  Future<void> deleteTask(String id) async {
    await _dio.delete('/api/tasks/$id');
  }

  /// Polls the server every [interval] and yields a [Task] whenever
  /// status / progress / error changes. Completes when the task reaches a
  /// terminal state.
  ///
  /// Polling (rather than SSE) is deliberate: it survives screen rotation,
  /// backgrounding, and flaky mobile networks without reconnect logic.
  Stream<Task> watchTask(
    String id, {
    Duration interval = const Duration(seconds: 1),
  }) async* {
    Task? last;
    while (true) {
      Task t;
      try {
        t = await getTask(id);
      } catch (e) {
        // Transient network error — surface and keep trying.
        // Give up after a long streak by rethrowing.
        rethrow;
      }

      final changed = last == null ||
          last.status != t.status ||
          last.progress.done != t.progress.done ||
          last.error != t.error ||
          last.outputSize != t.outputSize;

      if (changed) {
        last = t;
        yield t;
      } else {
        last = t;
      }

      if (t.status.isTerminal) return;
      await Future<void>.delayed(interval);
    }
  }

  /// Download the finished zip to [savePath]. Reports bytes received.
  Future<void> download({
    required String taskId,
    required String savePath,
    void Function(int received, int total)? onProgress,
    CancelToken? cancelToken,
  }) async {
    await _dio.download(
      '/api/tasks/$taskId/download',
      savePath,
      onReceiveProgress: onProgress,
      cancelToken: cancelToken,
      options: Options(
        receiveTimeout: Duration.zero, // large files, no per-chunk timeout
      ),
    );
  }

  /// Optional: subscribe to server-sent events for a task. The server exposes
  /// `GET /api/tasks/{id}/events`. Prefer [watchTask] unless you specifically
  /// want sub-second updates.
  Stream<Task> streamTaskSse(String id) {
    final controller = StreamController<Task>();
    StreamSubscription<String>? sub;

    Future<void> connect() async {
      try {
        final resp = await _dio.get<ResponseBody>(
          '/api/tasks/$id/events',
          options: Options(
            responseType: ResponseType.stream,
            headers: {'Accept': 'text/event-stream'},
            receiveTimeout: Duration.zero,
          ),
        );
        final stream = (resp.data!.stream)
            .transform(utf8.decoder as StreamTransformer<Uint8List, dynamic>)
            .transform(const LineSplitter());

        final buf = StringBuffer();
        sub = stream.listen(
          (line) {
            if (line.isEmpty) {
              final payload = buf.toString().trim();
              buf.clear();
              if (payload.isEmpty) return;
              try {
                final t =
                    Task.fromJson(jsonDecode(payload) as Map<String, dynamic>);
                controller.add(t);
                if (t.status.isTerminal) {
                  sub?.cancel();
                  if (!controller.isClosed) controller.close();
                }
              } catch (_) {/* ignore malformed frames */}
            } else if (line.startsWith('data:')) {
              buf.write(line.substring(5).trim());
            }
          },
          onError: (e) {
            controller.addError(e);
            if (!controller.isClosed) controller.close();
          },
          onDone: () {
            if (!controller.isClosed) controller.close();
          },
        );
      } catch (e) {
        controller.addError(e);
        if (!controller.isClosed) await controller.close();
      }
    }

    controller.onListen = connect;
    controller.onCancel = () async {
      await sub?.cancel();
    };
    return controller.stream;
  }
}
