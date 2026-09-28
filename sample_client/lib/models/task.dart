enum TaskStatus { pending, running, done, failed, cancelled, unknown }

TaskStatus parseStatus(String s) {
  switch (s) {
    case 'pending':
      return TaskStatus.pending;
    case 'running':
      return TaskStatus.running;
    case 'done':
      return TaskStatus.done;
    case 'failed':
      return TaskStatus.failed;
    case 'cancelled':
      return TaskStatus.cancelled;
    default:
      return TaskStatus.unknown;
  }
}

extension TaskStatusX on TaskStatus {
  String get label {
    switch (this) {
      case TaskStatus.pending:
        return 'Pending';
      case TaskStatus.running:
        return 'Running';
      case TaskStatus.done:
        return 'Done';
      case TaskStatus.failed:
        return 'Failed';
      case TaskStatus.cancelled:
        return 'Cancelled';
      case TaskStatus.unknown:
        return 'Unknown';
    }
  }

  bool get isTerminal =>
      this == TaskStatus.done ||
      this == TaskStatus.failed ||
      this == TaskStatus.cancelled;
}

class TaskProgress {
  final int total;
  final int done;
  final double percent;
  final String currentPage;

  const TaskProgress({
    this.total = 0,
    this.done = 0,
    this.percent = 0,
    this.currentPage = '',
  });

  factory TaskProgress.fromJson(Map<String, dynamic>? j) {
    if (j == null) return const TaskProgress();
    return TaskProgress(
      total: (j['total'] ?? 0) as int,
      done: (j['done'] ?? 0) as int,
      percent: ((j['percent'] ?? 0) as num).toDouble(),
      currentPage: (j['current_page'] ?? '') as String,
    );
  }
}

class Task {
  final String id;
  final TaskStatus status;
  final DateTime createdAt;
  final DateTime? startedAt;
  final DateTime? finishedAt;
  final String error;
  final TaskProgress progress;

  final String inputName;
  final int inputSize;
  final String outputName;
  final int outputSize;
  final String net;
  final int scale;

  Task({
    required this.id,
    required this.status,
    required this.createdAt,
    required this.startedAt,
    required this.finishedAt,
    required this.error,
    required this.progress,
    required this.inputName,
    required this.inputSize,
    required this.outputName,
    required this.outputSize,
    required this.net,
    required this.scale,
  });

  factory Task.fromJson(Map<String, dynamic> j) {
    DateTime? dt(Object? v) =>
        v is String && v.isNotEmpty ? DateTime.tryParse(v)?.toLocal() : null;

    return Task(
      id: j['id'] as String,
      status: parseStatus((j['status'] ?? '') as String),
      createdAt: dt(j['created_at']) ?? DateTime.now(),
      startedAt: dt(j['started_at']),
      finishedAt: dt(j['finished_at']),
      error: (j['error'] ?? '') as String,
      progress: TaskProgress.fromJson(j['progress'] as Map<String, dynamic>?),
      inputName: (j['input_name'] ?? '') as String,
      inputSize: (j['input_size'] ?? 0) as int,
      outputName: (j['output_name'] ?? '') as String,
      outputSize: (j['output_size'] ?? 0) as int,
      net: (j['net'] ?? '') as String,
      scale: (j['scale'] ?? 0) as int,
    );
  }
}
