import 'dart:async';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:open_filex/open_filex.dart';

import '../api/upscale_api.dart';
import '../models/task.dart';
import '../services/result_store.dart';

class TaskScreen extends StatefulWidget {
  final UpscaleApi api;
  final String taskId;
  const TaskScreen({super.key, required this.api, required this.taskId});

  @override
  State<TaskScreen> createState() => _TaskScreenState();
}

class _TaskScreenState extends State<TaskScreen> {
  Task? _task;
  StreamSubscription<Task>? _sub;
  String? _watchError;

  bool _downloading = false;
  double _dlProgress = 0;
  CancelToken? _dlCancel;
  String? _savedPath;

  @override
  void initState() {
    super.initState();
    _sub = widget.api.watchTask(widget.taskId).listen(
      (t) {
        if (!mounted) return;
        setState(() {
          _task = t;
          _watchError = null;
        });
      },
      onError: (e) {
        if (!mounted) return;
        setState(() => _watchError = '$e');
      },
    );
  }

  @override
  void dispose() {
    _sub?.cancel();
    _dlCancel?.cancel();
    super.dispose();
  }

  Future<void> _download() async {
    final task = _task;
    if (task == null) return;

    final dir = await resultStoreDir();
    final name =
        task.outputName.isNotEmpty ? task.outputName : '${task.id}.zip';
    final dest = File('${dir.path}${Platform.pathSeparator}$name');

    setState(() {
      _downloading = true;
      _dlProgress = 0;
      _savedPath = null;
    });
    final cancel = CancelToken();
    _dlCancel = cancel;

    try {
      await widget.api.download(
        taskId: task.id,
        savePath: dest.path,
        onProgress: (r, t) {
          if (!mounted || t <= 0) return;
          setState(() => _dlProgress = r / t);
        },
        cancelToken: cancel,
      );
      if (!mounted) return;
      setState(() => _savedPath = dest.path);
      _snack('Saved to ${dest.path}');
    } on DioException catch (e) {
      if (CancelToken.isCancel(e)) {
        _snack('Download cancelled');
      } else {
        _snack('Download failed: ${e.message}');
      }
    } catch (e) {
      _snack('Download failed: $e');
    } finally {
      _dlCancel = null;
      if (mounted) setState(() => _downloading = false);
    }
  }

  /// Save to a user-chosen location via SAF. No permission required on any
  /// Android version because the user explicitly authorises the destination.
  Future<void> _saveAs() async {
    final p = _savedPath;
    if (p == null) return;
    try {
      final target = await saveToPublicDownloads(
        sourcePath: p,
        suggestedName: p.split(Platform.pathSeparator).last,
      );
      if (target != null) _snack('Saved to $target');
    } catch (e) {
      _snack('Save failed: $e');
    }
  }

  Future<void> _open() async {
    final p = _savedPath;
    if (p == null) return;
    final res = await OpenFilex.open(p);
    if (res.type != ResultType.done) {
      _snack('Cannot open: ${res.message}');
    }
  }

  Future<void> _copyId() async {
    await Clipboard.setData(ClipboardData(text: widget.taskId));
    _snack('Task id copied');
  }

  void _snack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  @override
  Widget build(BuildContext context) {
    final t = _task;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Task'),
        actions: [
          IconButton(
            tooltip: 'Copy task id',
            onPressed: _copyId,
            icon: const Icon(Icons.tag),
          ),
        ],
      ),
      body: SafeArea(
        child: t == null
            ? const Center(child: CircularProgressIndicator())
            : ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  _Header(task: t),
                  const SizedBox(height: 16),
                  _ProgressCard(task: t),
                  if (t.error.isNotEmpty) ...[
                    const SizedBox(height: 12),
                    _ErrorCard(text: t.error),
                  ],
                  if (_watchError != null) ...[
                    const SizedBox(height: 12),
                    _ErrorCard(text: 'Connection: $_watchError'),
                  ],
                  const SizedBox(height: 24),
                  _MetaCard(task: t),
                  const SizedBox(height: 24),
                  if (_downloading)
                    _DownloadProgress(
                        progress: _dlProgress,
                        onCancel: () => _dlCancel?.cancel()),
                  if (t.status == TaskStatus.done) ...[
                    if (_savedPath == null)
                      FilledButton.icon(
                        onPressed: _downloading ? null : _download,
                        icon: const Icon(Icons.download),
                        label: const Text('Download result'),
                      )
                    else
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          FilledButton.icon(
                            onPressed: _open,
                            icon: const Icon(Icons.open_in_new),
                            label: const Text('Open'),
                          ),
                          const SizedBox(height: 8),
                          OutlinedButton.icon(
                            onPressed: _saveAs,
                            icon: const Icon(Icons.save_alt),
                            label: const Text('Save to…'),
                          ),
                          const SizedBox(height: 8),
                          OutlinedButton.icon(
                            onPressed: _download,
                            icon: const Icon(Icons.download),
                            label: const Text('Re-download'),
                          ),
                        ],
                      ),
                  ],
                ],
              ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------

class _Header extends StatelessWidget {
  final Task task;
  const _Header({required this.task});
  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          task.inputName.isEmpty ? task.id : task.inputName,
          style: Theme.of(context).textTheme.titleLarge,
        ),
        const SizedBox(height: 4),
        Text(
          task.id,
          style: Theme.of(context)
              .textTheme
              .bodySmall
              ?.copyWith(fontFamily: 'monospace'),
        ),
      ],
    );
  }
}

class _ProgressCard extends StatelessWidget {
  final Task task;
  const _ProgressCard({required this.task});

  @override
  Widget build(BuildContext context) {
    final p = task.progress;
    final active =
        task.status == TaskStatus.running || task.status == TaskStatus.pending;
    final value = p.total == 0 ? null : (p.done / p.total).clamp(0, 1);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    task.status.label,
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ),
                if (active)
                  const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
              ],
            ),
            const SizedBox(height: 12),
            // LinearProgressIndicator(value: value),
            const SizedBox(height: 8),
            Text(
              '${p.done}/${p.total}'
              '${p.currentPage.isEmpty ? '' : ' · ${p.currentPage}'}'
              '${p.percent > 0 ? ' · ${p.percent.toStringAsFixed(1)}%' : ''}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}

class _MetaCard extends StatelessWidget {
  final Task task;
  const _MetaCard({required this.task});

  @override
  Widget build(BuildContext context) {
    final rows = <(String, String)>[
      ('Model', task.net),
      ('Scale', '${task.scale}x'),
      ('Input size', _fmtBytes(task.inputSize)),
      if (task.outputSize > 0) ('Output size', _fmtBytes(task.outputSize)),
      ('Created', _fmtTime(task.createdAt)),
      if (task.startedAt != null) ('Started', _fmtTime(task.startedAt!)),
      if (task.finishedAt != null) ('Finished', _fmtTime(task.finishedAt!)),
    ];
    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        child: Column(
          children: [
            for (final (k, v) in rows)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 6),
                child: Row(
                  children: [
                    SizedBox(
                      width: 110,
                      child: Text(
                        k,
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ),
                    Expanded(child: SelectableText(v)),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _DownloadProgress extends StatelessWidget {
  final double progress;
  final VoidCallback onCancel;
  const _DownloadProgress({required this.progress, required this.onCancel});

  @override
  Widget build(BuildContext context) {
    final pct = (progress * 100).clamp(0, 100).toStringAsFixed(1);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Expanded(child: Text('Downloading…')),
                Text('$pct%'),
                IconButton(
                  onPressed: onCancel,
                  icon: const Icon(Icons.close),
                  tooltip: 'Cancel',
                ),
              ],
            ),
            const SizedBox(height: 4),
            LinearProgressIndicator(value: progress.clamp(0, 1)),
          ],
        ),
      ),
    );
  }
}

class _ErrorCard extends StatelessWidget {
  final String text;
  const _ErrorCard({required this.text});
  @override
  Widget build(BuildContext context) {
    final c = Theme.of(context).colorScheme;
    return Card(
      color: c.errorContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Text(text, style: TextStyle(color: c.onErrorContainer)),
      ),
    );
  }
}

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

String _fmtTime(DateTime d) {
  String p(int n) => n.toString().padLeft(2, '0');
  return '${d.year}-${p(d.month)}-${p(d.day)} '
      '${p(d.hour)}:${p(d.minute)}:${p(d.second)}';
}
