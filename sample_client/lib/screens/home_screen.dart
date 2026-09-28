import 'dart:async';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../api/upscale_api.dart';
import '../models/task.dart';
import 'task_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  static const _prefsKey = 'server_url';

  final _serverCtrl = TextEditingController();
  UpscaleApi? _api;
  Timer? _refreshTimer;

  bool _connecting = false;

  // upload state
  bool _uploading = false;
  double _uploadProgress = 0;
  String? _uploadName;
  CancelToken? _uploadCancel;

  // list state
  List<Task> _tasks = const [];
  String? _listError;

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  @override
  void dispose() {
    _refreshTimer?.cancel();
    _serverCtrl.dispose();
    _uploadCancel?.cancel();
    super.dispose();
  }

  Future<void> _bootstrap() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getString(_prefsKey);
    if (saved != null && saved.isNotEmpty) {
      _serverCtrl.text = saved;
      await _connect(saved);
    }
  }

  Future<void> _connect(String rawUrl) async {
    var url = rawUrl.trim();
    if (url.isEmpty) return;
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      url = 'http://$url';
    }
    url = url.replaceAll(RegExp(r'/+$'), '');

    setState(() => _connecting = true);
    final api = UpscaleApi(baseUrl: url);
    try {
      await api.health();
      await (await SharedPreferences.getInstance()).setString(_prefsKey, url);
      _api = api;
      _listError = null;
      _startRefresh();
      await _refresh();
      _snack('Connected to $url');
    } catch (e) {
      _api = null;
      _snack('Cannot reach $url: ${_short(e)}');
    } finally {
      if (mounted) setState(() => _connecting = false);
    }
  }

  void _startRefresh() {
    _refreshTimer?.cancel();
    _refreshTimer =
        Timer.periodic(const Duration(seconds: 4), (_) => _refresh());
  }

  Future<void> _refresh() async {
    final api = _api;
    if (api == null) return;
    try {
      final list = await api.listTasks();
      list.sort((a, b) => b.createdAt.compareTo(a.createdAt));
      if (!mounted) return;
      setState(() {
        _tasks = list;
        _listError = null;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _listError = _short(e));
    }
  }

  Future<void> _pickAndUpload() async {
    final api = _api;
    if (api == null) return;

    // ---- Android permission note -------------------------------------------
    // file_picker dispatches Intent.ACTION_OPEN_DOCUMENT (the Storage Access
    // Framework). That is why there is NO permission_handler call anywhere in
    // this app: the user explicitly picking a file grants our app access to
    // exactly that URI. This works identically on Android 4.4 through 15+,
    // including scoped-storage devices.
    // ------------------------------------------------------------------------
    final picked = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: const ['cbz', 'zip'],
      allowMultiple: false,
      withData: false,
    );
    if (picked == null || picked.files.isEmpty) return;

    final f = picked.files.single;
    final path = f.path;
    if (path == null) {
      _snack('Could not read the picked file.');
      return;
    }

    final size = await File(path).length();

    setState(() {
      _uploading = true;
      _uploadProgress = 0;
      _uploadName = f.name;
    });
    final cancel = CancelToken();
    _uploadCancel = cancel;

    try {
      final res = await api.upload(
        filePath: path,
        fileName: f.name,
        onProgress: (sent, total) {
          if (!mounted || total <= 0) return;
          setState(() => _uploadProgress = sent / total);
        },
        cancelToken: cancel,
      );

      _snack('Uploaded ${f.name} (${_fmtBytes(size)})');
      await _refresh();
      if (!mounted) return;
      await Navigator.of(context).push(MaterialPageRoute(
        builder: (_) => TaskScreen(api: api, taskId: res.taskId),
      ));
      _refresh();
    } on DioException catch (e) {
      if (CancelToken.isCancel(e)) {
        _snack('Upload cancelled');
      } else {
        _snack('Upload failed: ${_shortDio(e)}');
      }
    } catch (e) {
      _snack('Upload failed: $e');
    } finally {
      _uploadCancel = null;
      if (mounted) {
        setState(() {
          _uploading = false;
          _uploadProgress = 0;
          _uploadName = null;
        });
      }
    }
  }

  void _snack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  @override
  Widget build(BuildContext context) {
    final api = _api;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Upscale'),
        actions: [
          IconButton(
            tooltip: 'Refresh',
            onPressed: api == null ? null : _refresh,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            _ServerBar(
              controller: _serverCtrl,
              connected: api != null,
              connecting: _connecting,
              onConnect: () => _connect(_serverCtrl.text),
            ),
            if (_listError != null) _ErrorBanner(text: _listError!),
            if (_uploading)
              _UploadBanner(
                name: _uploadName ?? '',
                progress: _uploadProgress,
                onCancel: () => _uploadCancel?.cancel(),
              ),
            Expanded(
              child: api == null
                  ? const _EmptyState(
                      icon: Icons.cloud_off,
                      title: 'Not connected',
                      subtitle: 'Enter your server URL above and tap Connect.',
                    )
                  : _tasks.isEmpty
                      ? const _EmptyState(
                          icon: Icons.inbox_outlined,
                          title: 'No tasks yet',
                          subtitle: 'Tap Upload to send a .cbz to the server.',
                        )
                      : RefreshIndicator(
                          onRefresh: _refresh,
                          child: ListView.separated(
                            itemCount: _tasks.length,
                            separatorBuilder: (_, __) =>
                                const Divider(height: 1),
                            itemBuilder: (_, i) => _TaskTile(
                              task: _tasks[i],
                              onOpen: () async {
                                await Navigator.of(context).push(
                                  MaterialPageRoute(
                                    builder: (_) => TaskScreen(
                                      api: api,
                                      taskId: _tasks[i].id,
                                    ),
                                  ),
                                );
                                _refresh();
                              },
                              onDelete: () async {
                                try {
                                  await api.deleteTask(_tasks[i].id);
                                } catch (_) {}
                                _refresh();
                              },
                            ),
                          ),
                        ),
            ),
          ],
        ),
      ),
      floatingActionButton: api == null
          ? null
          : FloatingActionButton.extended(
              onPressed: _uploading ? null : _pickAndUpload,
              icon: const Icon(Icons.upload_file),
              label: Text(_uploading ? 'Uploading…' : 'Upload'),
            ),
    );
  }
}

// ---------------------------------------------------------------------------
// Small widgets
// ---------------------------------------------------------------------------

class _ServerBar extends StatelessWidget {
  final TextEditingController controller;
  final bool connected;
  final bool connecting;
  final VoidCallback onConnect;

  const _ServerBar({
    required this.controller,
    required this.connected,
    required this.connecting,
    required this.onConnect,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 12, 12, 6),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: controller,
              keyboardType: TextInputType.url,
              autocorrect: false,
              decoration: InputDecoration(
                isDense: true,
                hintText: 'http://192.168.1.10:8080',
                prefixIcon: Icon(
                  connected ? Icons.cloud_done : Icons.cloud_queue,
                  size: 18,
                ),
                border: const OutlineInputBorder(),
              ),
              onSubmitted: (_) => onConnect(),
            ),
          ),
          const SizedBox(width: 8),
          FilledButton(
            onPressed: connecting ? null : onConnect,
            child: connecting
                ? const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Text('Connect'),
          ),
        ],
      ),
    );
  }
}

class _UploadBanner extends StatelessWidget {
  final String name;
  final double progress;
  final VoidCallback onCancel;
  const _UploadBanner({
    required this.name,
    required this.progress,
    required this.onCancel,
  });

  @override
  Widget build(BuildContext context) {
    final pct = (progress * 100).clamp(0, 100).toStringAsFixed(1);
    return Material(
      color: Theme.of(context).colorScheme.surfaceContainerHighest,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 10, 8, 10),
        child: Row(
          children: [
            const Icon(Icons.upload, size: 20),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Uploading $name',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(fontWeight: FontWeight.w500),
                  ),
                  const SizedBox(height: 6),
                  LinearProgressIndicator(value: progress.clamp(0, 1)),
                  const SizedBox(height: 4),
                  Text('$pct%', style: Theme.of(context).textTheme.bodySmall),
                ],
              ),
            ),
            IconButton(
              tooltip: 'Cancel',
              onPressed: onCancel,
              icon: const Icon(Icons.close),
            ),
          ],
        ),
      ),
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  final String text;
  const _ErrorBanner({required this.text});
  @override
  Widget build(BuildContext context) {
    final c = Theme.of(context).colorScheme;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      color: c.errorContainer,
      child: Text(text, style: TextStyle(color: c.onErrorContainer)),
    );
  }
}

class _EmptyState extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  const _EmptyState({
    required this.icon,
    required this.title,
    required this.subtitle,
  });
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 48, color: Theme.of(context).disabledColor),
            const SizedBox(height: 12),
            Text(title, style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 6),
            Text(
              subtitle,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}

class _TaskTile extends StatelessWidget {
  final Task task;
  final VoidCallback onOpen;
  final VoidCallback onDelete;

  const _TaskTile({
    required this.task,
    required this.onOpen,
    required this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    final progress = task.progress.percent / 100;
    final isActive =
        task.status == TaskStatus.running || task.status == TaskStatus.pending;

    return ListTile(
      onTap: onOpen,
      leading: _StatusDot(status: task.status),
      title: Text(
        task.inputName.isEmpty ? task.id : task.inputName,
        maxLines: 1,
        overflow: TextOverflow.ellipsis,
      ),
      subtitle: Padding(
        padding: const EdgeInsets.only(top: 4),
        child: isActive
            ? Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  LinearProgressIndicator(
                    value: progress <= 0 ? null : progress.clamp(0, 1),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '${task.status.label} · '
                    '${task.progress.done}/${task.progress.total}'
                    '${task.progress.currentPage.isEmpty ? '' : ' · ${task.progress.currentPage}'}',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              )
            : Text(
                '${task.status.label}'
                '${task.status == TaskStatus.done ? ' · ${_fmtBytes(task.outputSize)}' : ''}'
                '${task.error.isEmpty ? '' : ' · ${task.error}'}',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
      ),
      trailing: IconButton(
        tooltip: isActive ? 'Cancel' : 'Delete',
        icon: Icon(isActive ? Icons.cancel_outlined : Icons.delete_outline),
        onPressed: onDelete,
      ),
    );
  }
}

class _StatusDot extends StatelessWidget {
  final TaskStatus status;
  const _StatusDot({required this.status});
  @override
  Widget build(BuildContext context) {
    late final Color color;
    switch (status) {
      case TaskStatus.pending:
        color = Colors.amber;
        break;
      case TaskStatus.running:
        color = Colors.lightBlue;
        break;
      case TaskStatus.done:
        color = Colors.green;
        break;
      case TaskStatus.failed:
        color = Colors.red;
        break;
      case TaskStatus.cancelled:
        color = Colors.grey;
        break;
      case TaskStatus.unknown:
        color = Colors.grey;
        break;
    }
    return Container(
      width: 12,
      height: 12,
      decoration: BoxDecoration(color: color, shape: BoxShape.circle),
    );
  }
}

// ---------------------------------------------------------------------------
// formatting helpers
// ---------------------------------------------------------------------------

String _fmtBytes(int n) {
  if (n <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  var d = n.toDouble();
  var i = 0;
  while (d >= 1024 && i < units.length - 1) {
    d /= 1024;
    i++;
  }
  return '${d.toStringAsFixed(d >= 100 || i == 0 ? 0 : 1)} ${units[i]}';
}

String _short(Object e) {
  final s = e.toString();
  return s.length > 160 ? '${s.substring(0, 160)}…' : s;
}

String _shortDio(DioException e) {
  final msg = e.response?.data is Map
      ? (e.response!.data['error'] ?? e.message).toString()
      : (e.message ?? e.toString());
  return _short(msg);
}
