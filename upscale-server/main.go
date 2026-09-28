package main

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/rand"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	_ "image/gif"

	_ "golang.org/x/image/webp"
)

// ---------------------------------------------------------------------------
// Embedded Real-ESRGAN binary + models
// ---------------------------------------------------------------------------

//go:embed assets/realesrgan-ncnn-vulkan
var embeddedBin []byte

//go:embed assets/models/*
var embeddedModels embed.FS

// Bump when embedded assets change, forces re-extraction to cache dir.
const assetsVersion = "v1"

func assetCacheDir() (string, error) {
	root, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "upscale-server", assetsVersion), nil
}

func writeFileAtomic(path string, data []byte, mode os.FileMode) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, mode); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func ensureAssets() (binPath, modelsDir string, err error) {
	root, err := assetCacheDir()
	if err != nil {
		return "", "", err
	}
	binPath = filepath.Join(root, "realesrgan-ncnn-vulkan")
	modelsDir = filepath.Join(root, "models")

	if info, statErr := os.Stat(binPath); statErr == nil && info.Size() == int64(len(embeddedBin)) {
		if _, statErr := os.Stat(modelsDir); statErr == nil {
			return binPath, modelsDir, nil
		}
	}
	log.Printf("extracting embedded assets to %s", root)
	if err := os.MkdirAll(root, 0o755); err != nil {
		return "", "", err
	}
	if err := writeFileAtomic(binPath, embeddedBin, 0o755); err != nil {
		return "", "", err
	}
	if err := os.MkdirAll(modelsDir, 0o755); err != nil {
		return "", "", err
	}
	entries, err := embeddedModels.ReadDir("assets/models")
	if err != nil {
		return "", "", err
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		data, err := embeddedModels.ReadFile("assets/models/" + e.Name())
		if err != nil {
			return "", "", err
		}
		if err := writeFileAtomic(filepath.Join(modelsDir, e.Name()), data, 0o644); err != nil {
			return "", "", err
		}
	}
	return binPath, modelsDir, nil
}

// ---------------------------------------------------------------------------
// Vulkan upscaler (per-image subprocess)
// ---------------------------------------------------------------------------

type VulkanUpscaler struct {
	binPath, modelsDir string
	net                string
	scale, tile, gpu   int
}

func (u *VulkanUpscaler) Upscale(img image.Image) (image.Image, error) {
	tmpDir, err := os.MkdirTemp("", "upscale-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)

	inPath := filepath.Join(tmpDir, "in.png")
	outPath := filepath.Join(tmpDir, "out.png")

	if err := writePNG(inPath, img); err != nil {
		return nil, err
	}
	args := []string{
		"-i", inPath, "-o", outPath,
		"-n", u.net, "-s", strconv.Itoa(u.scale),
		"-m", u.modelsDir, "-g", strconv.Itoa(u.gpu),
	}
	if u.tile > 0 {
		args = append(args, "-t", strconv.Itoa(u.tile))
	}
	cmd := exec.Command(u.binPath, args...)
	cmd.Dir = tmpDir
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("realesrgan: %w\n%s", err, lastLines(stderr.String(), 6))
	}
	f, err := os.Open(outPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	out, _, err := image.Decode(f)
	return out, err
}

func writePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func lastLines(s string, n int) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) <= n {
		return strings.Join(lines, "\n")
	}
	return strings.Join(lines[len(lines)-n:], "\n")
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

type Config struct {
	Addr        string
	DataDir     string
	Net         string
	Scale       int
	TileSize    int
	GPU         int
	JPEGQuality int
	Workers     int
	MaxUploadMB int64
}

func parseConfig() Config {
	var c Config
	flag.StringVar(&c.Addr, "addr", ":8080", "HTTP listen address")
	flag.StringVar(&c.DataDir, "data", "./data", "persistent data directory")
	flag.StringVar(&c.Net, "net", "realesrgan-x4plus-anime", "default model")
	flag.IntVar(&c.Scale, "scale", 4, "default scale factor")
	flag.IntVar(&c.TileSize, "tile", 0, "ncnn tile size (0 = auto)")
	flag.IntVar(&c.GPU, "gpu", 0, "Vulkan device id (-1 = CPU)")
	flag.IntVar(&c.JPEGQuality, "jpeg-quality", 95, "JPEG output quality")
	flag.IntVar(&c.Workers, "workers", 1, "concurrent upscale workers")
	flag.Int64Var(&c.MaxUploadMB, "max-upload-mb", 2048, "max upload size (MB)")
	flag.Parse()
	return c
}

// ---------------------------------------------------------------------------
// Task model
// ---------------------------------------------------------------------------

type TaskStatus string

const (
	StatusPending   TaskStatus = "pending"
	StatusRunning   TaskStatus = "running"
	StatusDone      TaskStatus = "done"
	StatusFailed    TaskStatus = "failed"
	StatusCancelled TaskStatus = "cancelled"
)

type Progress struct {
	Total       int     `json:"total"`
	Done        int     `json:"done"`
	Percent     float64 `json:"percent"`
	CurrentPage string  `json:"current_page,omitempty"`
}

type Task struct {
	ID         string     `json:"id"`
	Status     TaskStatus `json:"status"`
	CreatedAt  time.Time  `json:"created_at"`
	StartedAt  *time.Time `json:"started_at,omitempty"`
	FinishedAt *time.Time `json:"finished_at,omitempty"`
	Error      string     `json:"error,omitempty"`
	Progress   Progress   `json:"progress"`

	InputName  string `json:"input_name"`
	InputSize  int64  `json:"input_size"`
	OutputName string `json:"output_name,omitempty"`
	OutputSize int64  `json:"output_size,omitempty"`

	Net   string `json:"net"`
	Scale int    `json:"scale"`
}

// ---------------------------------------------------------------------------
// Task manager (in-memory map + on-disk persistence)
// ---------------------------------------------------------------------------

type TaskManager struct {
	mu      sync.RWMutex
	tasks   map[string]*Task
	cancels map[string]context.CancelFunc
	dataDir string
}

func NewTaskManager(dataDir string) (*TaskManager, error) {
	tm := &TaskManager{
		tasks:   map[string]*Task{},
		cancels: map[string]context.CancelFunc{},
		dataDir: dataDir,
	}
	if err := os.MkdirAll(filepath.Join(dataDir, "tasks"), 0o755); err != nil {
		return nil, err
	}
	if err := tm.loadAll(); err != nil {
		return nil, err
	}
	return tm, nil
}

func (tm *TaskManager) loadAll() error {
	root := filepath.Join(tm.dataDir, "tasks")
	dirs, err := os.ReadDir(root)
	if err != nil {
		return err
	}
	for _, d := range dirs {
		if !d.IsDir() {
			continue
		}
		b, err := os.ReadFile(filepath.Join(root, d.Name(), "task.json"))
		if err != nil {
			continue
		}
		var t Task
		if err := json.Unmarshal(b, &t); err != nil {
			continue
		}
		// Server crashed mid-run: mark failed (client can re-submit).
		if t.Status == StatusRunning {
			t.Status = StatusFailed
			t.Error = "server restarted"
			now := time.Now()
			t.FinishedAt = &now
		}
		tm.tasks[t.ID] = &t
	}
	return nil
}

func newID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func (tm *TaskManager) TaskDir(id string) string {
	return filepath.Join(tm.dataDir, "tasks", id)
}
func (tm *TaskManager) InputPath(id string) string {
	return filepath.Join(tm.TaskDir(id), "input.zip")
}
func (tm *TaskManager) OutputPath(id string) string {
	return filepath.Join(tm.TaskDir(id), "output.zip")
}

func (tm *TaskManager) Create(inputName string, inputSize int64, net string, scale int) (*Task, error) {
	t := &Task{
		ID:        newID(),
		Status:    StatusPending,
		CreatedAt: time.Now().UTC(),
		InputName: inputName,
		InputSize: inputSize,
		Net:       net,
		Scale:     scale,
	}
	if err := os.MkdirAll(tm.TaskDir(t.ID), 0o755); err != nil {
		return nil, err
	}
	tm.mu.Lock()
	tm.tasks[t.ID] = t
	tm.mu.Unlock()
	if err := tm.save(t); err != nil {
		return nil, err
	}
	return t, nil
}

func (tm *TaskManager) Get(id string) (*Task, bool) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	t, ok := tm.tasks[id]
	if !ok {
		return nil, false
	}
	cp := *t
	return &cp, true
}

func (tm *TaskManager) List() []*Task {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	out := make([]*Task, 0, len(tm.tasks))
	for _, t := range tm.tasks {
		cp := *t
		out = append(out, &cp)
	}
	return out
}

func (tm *TaskManager) Update(id string, fn func(*Task)) error {
	tm.mu.Lock()
	t, ok := tm.tasks[id]
	if !ok {
		tm.mu.Unlock()
		return fmt.Errorf("task %s not found", id)
	}
	fn(t)
	cp := *t
	tm.mu.Unlock()
	return tm.save(&cp)
}

func (tm *TaskManager) SetCancel(id string, fn context.CancelFunc) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	tm.cancels[id] = fn
}

func (tm *TaskManager) Cancel(id string) bool {
	tm.mu.RLock()
	fn := tm.cancels[id]
	tm.mu.RUnlock()
	if fn != nil {
		fn()
		return true
	}
	return false
}

func (tm *TaskManager) Delete(id string) error {
	tm.mu.Lock()
	delete(tm.tasks, id)
	fn := tm.cancels[id]
	delete(tm.cancels, id)
	tm.mu.Unlock()
	if fn != nil {
		fn()
	}
	return os.RemoveAll(tm.TaskDir(id))
}

func (tm *TaskManager) save(t *Task) error {
	b, err := json.MarshalIndent(t, "", "  ")
	if err != nil {
		return err
	}
	tmp := filepath.Join(tm.TaskDir(t.ID), "task.json.tmp")
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(tm.TaskDir(t.ID), "task.json"))
}

// ---------------------------------------------------------------------------
// Server + worker pool
// ---------------------------------------------------------------------------

type Server struct {
	cfg       Config
	tm        *TaskManager
	queue     chan string
	binPath   string
	modelsDir string
}

func NewServer(cfg Config) (*Server, error) {
	tm, err := NewTaskManager(cfg.DataDir)
	if err != nil {
		return nil, err
	}
	binPath, modelsDir, err := ensureAssets()
	if err != nil {
		return nil, err
	}
	return &Server{
		cfg:       cfg,
		tm:        tm,
		queue:     make(chan string, 256),
		binPath:   binPath,
		modelsDir: modelsDir,
	}, nil
}

func (s *Server) StartWorkers(ctx context.Context) {
	for i := 0; i < s.cfg.Workers; i++ {
		go s.worker(ctx, i)
	}
	// Re-queue any pending tasks from a prior run.
	for _, t := range s.tm.List() {
		if t.Status == StatusPending {
			if _, err := os.Stat(s.tm.InputPath(t.ID)); err == nil {
				s.queue <- t.ID
			}
		}
	}
}

func (s *Server) worker(ctx context.Context, n int) {
	log.Printf("worker %d started", n)
	for {
		select {
		case <-ctx.Done():
			return
		case id := <-s.queue:
			s.processTask(ctx, id)
		}
	}
}

func (s *Server) processTask(parent context.Context, id string) {
	t, ok := s.tm.Get(id)
	if !ok {
		return
	}
	if t.Status != StatusPending {
		return
	}

	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	s.tm.SetCancel(id, cancel)

	now := time.Now().UTC()
	_ = s.tm.Update(id, func(x *Task) {
		x.Status = StatusRunning
		x.StartedAt = &now
	})

	fail := func(err error) {
		log.Printf("[%s] failed: %v", id, err)
		now := time.Now().UTC()
		_ = s.tm.Update(id, func(x *Task) {
			x.Status = StatusFailed
			x.Error = err.Error()
			x.FinishedAt = &now
		})
	}

	r, err := zip.OpenReader(s.tm.InputPath(id))
	if err != nil {
		fail(fmt.Errorf("open input: %w", err))
		return
	}
	defer r.Close()

	totalImages := 0
	for _, f := range r.File {
		if !f.FileInfo().IsDir() && isImageEntry(f.Name) {
			totalImages++
		}
	}
	if totalImages == 0 {
		fail(errors.New("no images found in archive"))
		return
	}
	_ = s.tm.Update(id, func(x *Task) {
		x.Progress.Total = totalImages
	})

	out, err := os.Create(s.tm.OutputPath(id))
	if err != nil {
		fail(fmt.Errorf("create output: %w", err))
		return
	}
	defer out.Close()
	zw := zip.NewWriter(out)
	defer zw.Close()

	up := &VulkanUpscaler{
		binPath:   s.binPath,
		modelsDir: s.modelsDir,
		net:       t.Net,
		scale:     t.Scale,
		tile:      s.cfg.TileSize,
		gpu:       s.cfg.GPU,
	}

	renameMap, extChanges := buildRenameMaps(r.File)
	done := 0

	for _, f := range r.File {
		select {
		case <-ctx.Done():
			now := time.Now().UTC()
			_ = s.tm.Update(id, func(x *Task) {
				x.Status = StatusCancelled
				x.Error = "cancelled"
				x.FinishedAt = &now
			})
			_ = os.Remove(s.tm.OutputPath(id))
			return
		default:
		}

		raw, err := readZipEntry(f)
		if err != nil {
			fail(fmt.Errorf("read %s: %w", f.Name, err))
			return
		}

		if f.FileInfo().IsDir() || !isImageEntry(f.Name) {
			if err := writeZipEntry(zw, f.Name, f.Mode(), f.Modified, rewriteMeta(f.Name, raw, renameMap, extChanges)); err != nil {
				fail(err)
				return
			}
			continue
		}

		// Upscale
		img, err := decodeImage(raw)
		if err != nil {
			// passthrough
			if err := writeZipEntry(zw, f.Name, f.Mode(), f.Modified, raw); err != nil {
				fail(err)
				return
			}
		} else {
			result, err := up.Upscale(img)
			if err != nil {
				log.Printf("[%s] upscale %s failed: %v (passthrough)", id, f.Name, err)
				if err := writeZipEntry(zw, f.Name, f.Mode(), f.Modified, raw); err != nil {
					fail(err)
					return
				}
			} else {
				outName, data, err := encodeOutput(f.Name, result, s.cfg.JPEGQuality)
				if err != nil {
					fail(err)
					return
				}
				if err := writeZipEntry(zw, outName, f.Mode(), f.Modified, data); err != nil {
					fail(err)
					return
				}
			}
		}

		done++
		_ = s.tm.Update(id, func(x *Task) {
			x.Progress.Done = done
			x.Progress.Percent = float64(done) / float64(x.Progress.Total) * 100
			x.Progress.CurrentPage = f.Name
		})
		log.Printf("[%s] %d/%d  %s", id, done, totalImages, f.Name)
	}

	// Determine output size
	if fi, err := os.Stat(s.tm.OutputPath(id)); err == nil {
		_ = s.tm.Update(id, func(x *Task) {
			x.OutputSize = fi.Size()
		})
	}

	now = time.Now().UTC()
	_ = s.tm.Update(id, func(x *Task) {
		x.Status = StatusDone
		x.FinishedAt = &now
		x.OutputName = outputZipName(t.InputName)
		x.Progress.Percent = 100
		x.Progress.CurrentPage = ""
	})
	log.Printf("[%s] done (%d pages)", id, done)
}

// ---------------------------------------------------------------------------
// Zip / image helpers (kept from original)
// ---------------------------------------------------------------------------

func isImageEntry(name string) bool {
	for _, part := range strings.Split(name, "/") {
		if part == "__MACOSX" || strings.HasPrefix(part, "._") {
			return false
		}
	}
	switch strings.ToLower(filepath.Ext(name)) {
	case ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif":
		return true
	}
	return false
}

func isTextMetadata(name string) bool {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".json", ".xml", ".txt", ".opf", ".comicinfo",
		".yaml", ".yml", ".toml", ".csv", ".md", ".html", ".htm":
		return true
	}
	return false
}

func outputExtFor(origExt string) string {
	if strings.EqualFold(origExt, ".webp") {
		return ".jpg"
	}
	return origExt
}

func predictOutputName(origName string) string {
	ext := filepath.Ext(origName)
	newExt := outputExtFor(ext)
	if newExt == ext {
		return origName
	}
	return strings.TrimSuffix(origName, ext) + newExt
}

func buildRenameMaps(files []*zip.File) (renameMap, extChanges map[string]string) {
	renameMap = map[string]string{}
	extChanges = map[string]string{}
	for _, f := range files {
		if f.FileInfo().IsDir() || !isImageEntry(f.Name) {
			continue
		}
		oldName := f.Name
		newName := predictOutputName(oldName)
		if newName == oldName {
			continue
		}
		renameMap[oldName] = newName
		oe, ne := filepath.Ext(oldName), filepath.Ext(newName)
		if oe != ne {
			extChanges[oe] = ne
		}
	}
	return
}

func isWordByte(b byte) bool {
	switch {
	case b >= 'a' && b <= 'z', b >= 'A' && b <= 'Z', b >= '0' && b <= '9':
		return true
	case b == '_', b == '-':
		return true
	}
	return false
}

func replaceExtSmart(content []byte, oldExt, newExt string) []byte {
	oldB, newB := []byte(oldExt), []byte(newExt)
	var out []byte
	i := 0
	for i < len(content) {
		idx := bytes.Index(content[i:], oldB)
		if idx < 0 {
			out = append(out, content[i:]...)
			break
		}
		abs := i + idx
		end := abs + len(oldB)
		if end < len(content) && isWordByte(content[end]) {
			out = append(out, content[i:end]...)
			i = end
			continue
		}
		out = append(out, content[i:abs]...)
		out = append(out, newB...)
		i = end
	}
	return out
}

func rewriteMeta(name string, raw []byte, renameMap, extChanges map[string]string) []byte {
	if !isTextMetadata(name) {
		return raw
	}
	out := raw
	for old, new := range renameMap {
		out = bytes.ReplaceAll(out, []byte(old), []byte(new))
	}
	for oe, ne := range extChanges {
		out = replaceExtSmart(out, oe, ne)
	}
	return out
}

func readZipEntry(f *zip.File) ([]byte, error) {
	rc, err := f.Open()
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	return io.ReadAll(rc)
}

func writeZipEntry(zw *zip.Writer, name string, mode os.FileMode, mod time.Time, data []byte) error {
	hdr := &zip.FileHeader{Name: name, Method: zip.Deflate, Modified: mod}
	hdr.SetMode(mode)
	w, err := zw.CreateHeader(hdr)
	if err != nil {
		return err
	}
	if len(data) == 0 {
		return nil
	}
	_, err = w.Write(data)
	return err
}

func decodeImage(raw []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(raw))
	return img, err
}

func encodeOutput(origName string, img image.Image, quality int) (string, []byte, error) {
	origExt := filepath.Ext(origName)
	outExt := outputExtFor(origExt)
	outName := strings.TrimSuffix(origName, origExt) + outExt
	var buf bytes.Buffer
	switch strings.ToLower(outExt) {
	case ".jpg", ".jpeg":
		if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: quality}); err != nil {
			return "", nil, err
		}
	case ".png":
		if err := png.Encode(&buf, img); err != nil {
			return "", nil, err
		}
	default:
		if err := png.Encode(&buf, img); err != nil {
			return "", nil, err
		}
	}
	return outName, buf.Bytes(), nil
}

func outputZipName(input string) string {
	base := strings.TrimSuffix(filepath.Base(input), filepath.Ext(input))
	return base + ".upscaled.zip"
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

func (s *Server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", s.handleHealth)
	mux.HandleFunc("POST /api/tasks", s.handleCreateTask)
	mux.HandleFunc("GET /api/tasks", s.handleListTasks)
	mux.HandleFunc("GET /api/tasks/{id}", s.handleGetTask)
	mux.HandleFunc("GET /api/tasks/{id}/events", s.handleTaskEvents)
	mux.HandleFunc("GET /api/tasks/{id}/download", s.handleDownload)
	mux.HandleFunc("DELETE /api/tasks/{id}", s.handleDeleteTask)
	return withCORS(mux)
}

func withCORS(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		h.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func httpError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]any{"status": "ok", "time": time.Now().UTC()})
}

func (s *Server) handleCreateTask(w http.ResponseWriter, r *http.Request) {
	maxBytes := s.cfg.MaxUploadMB << 20
	r.Body = http.MaxBytesReader(w, r.Body, maxBytes)

	if err := r.ParseMultipartForm(32 << 20); err != nil {
		httpError(w, http.StatusBadRequest, "invalid multipart: "+err.Error())
		return
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		httpError(w, http.StatusBadRequest, "missing 'file' field")
		return
	}
	defer file.Close()

	if !isArchive(header.Filename) {
		httpError(w, http.StatusBadRequest, "only .zip / .cbz are supported")
		return
	}

	net := r.FormValue("net")
	if net == "" {
		net = s.cfg.Net
	}
	scale := s.cfg.Scale
	if v := r.FormValue("scale"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 || n > 8 {
			httpError(w, http.StatusBadRequest, "invalid scale")
			return
		}
		scale = n
	}
	// Sanity: model must exist
	paramPath := filepath.Join(s.modelsDir, net+".param")
	if _, err := os.Stat(paramPath); err != nil {
		httpError(w, http.StatusBadRequest, "unknown model: "+net)
		return
	}

	t, err := s.tm.Create(header.Filename, header.Size, net, scale)
	if err != nil {
		httpError(w, http.StatusInternalServerError, err.Error())
		return
	}

	dst, err := os.Create(s.tm.InputPath(t.ID))
	if err != nil {
		_ = s.tm.Delete(t.ID)
		httpError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if _, err := io.Copy(dst, file); err != nil {
		dst.Close()
		_ = s.tm.Delete(t.ID)
		httpError(w, http.StatusInternalServerError, "saving upload: "+err.Error())
		return
	}
	dst.Close()

	s.queue <- t.ID

	writeJSON(w, http.StatusAccepted, map[string]any{
		"task_id":    t.ID,
		"status":     t.Status,
		"status_url": "/api/tasks/" + t.ID,
	})
}

func (s *Server) handleListTasks(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]any{"tasks": s.tm.List()})
}

func (s *Server) handleGetTask(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	t, ok := s.tm.Get(id)
	if !ok {
		httpError(w, http.StatusNotFound, "task not found")
		return
	}
	writeJSON(w, 200, t)
}

func (s *Server) handleTaskEvents(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if _, ok := s.tm.Get(id); !ok {
		httpError(w, http.StatusNotFound, "task not found")
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		httpError(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	ticker := time.NewTicker(700 * time.Millisecond)
	defer ticker.Stop()

	last := ""
	for {
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
			t, ok := s.tm.Get(id)
			if !ok {
				fmt.Fprint(w, "event: error\ndata: task deleted\n\n")
				flusher.Flush()
				return
			}
			b, _ := json.Marshal(t)
			if string(b) != last {
				fmt.Fprintf(w, "data: %s\n\n", b)
				flusher.Flush()
				last = string(b)
			}
			switch t.Status {
			case StatusDone, StatusFailed, StatusCancelled:
				return
			}
		}
	}
}

func (s *Server) handleDownload(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	t, ok := s.tm.Get(id)
	if !ok {
		httpError(w, http.StatusNotFound, "task not found")
		return
	}
	if t.Status != StatusDone {
		httpError(w, http.StatusConflict, "task not done yet: "+string(t.Status))
		return
	}
	out := s.tm.OutputPath(id)
	fi, err := os.Stat(out)
	if err != nil {
		httpError(w, http.StatusInternalServerError, "output missing")
		return
	}
	name := t.OutputName
	if name == "" {
		name = outputZipName(t.InputName)
	}
	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename="%s"`, name))
	http.ServeContent(w, r, name, fi.ModTime(), mustOpen(out))
}

func mustOpen(p string) *os.File {
	f, err := os.Open(p)
	if err != nil {
		panic(err)
	}
	return f
}

func (s *Server) handleDeleteTask(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	t, ok := s.tm.Get(id)
	if !ok {
		httpError(w, http.StatusNotFound, "task not found")
		return
	}
	// Cancel if still running; otherwise just delete.
	switch t.Status {
	case StatusPending, StatusRunning:
		s.tm.Cancel(id)
		_ = s.tm.Update(id, func(x *Task) {
			x.Status = StatusCancelled
			now := time.Now().UTC()
			x.FinishedAt = &now
		})
	}
	if err := s.tm.Delete(id); err != nil {
		httpError(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

func main() {
	log.SetFlags(log.Ltime)
	cfg := parseConfig()

	srv, err := NewServer(cfg)
	if err != nil {
		log.Fatalf("init: %v", err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	srv.StartWorkers(ctx)

	httpSrv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           srv.routes(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	go func() {
		log.Printf("listening on %s (data dir: %s)", cfg.Addr, cfg.DataDir)
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("http: %v", err)
		}
	}()

	<-ctx.Done()
	log.Println("shutting down…")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = httpSrv.Shutdown(shutdownCtx)
}

// isArchive reports whether a filename looks like a zip/cbz we can open.
func isArchive(name string) bool {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".zip", ".cbz":
		return true
	}
	return false
}
