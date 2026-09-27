package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	xdraw "golang.org/x/image/draw"
	_ "golang.org/x/image/webp"

	_ "image/gif"
)

//go:embed assets/realesrgan-ncnn-vulkan
var embeddedBin []byte

//go:embed assets/models/*
var embeddedModels embed.FS

const (
	// assetsVersion 用于区分不同版本的内嵌资源。替换了模型或二进制之后
	// 把这个值改成 v2、v3…… 可以强制程序重新释放到缓存目录。
	assetsVersion = "v1"

	// binName 是 realesrgan-ncnn-vulkan 释放到缓存目录后的文件名。
	binName = "realesrgan-ncnn-vulkan"

	// 预览目录下的子目录名。保持 ASCII 避免跨平台兼容性问题。
	dirOriginal = "original" // 原图
	dirUpscaled = "upscaled" // 放大图
	dirCompare  = "compare"  // 并排对比图
)

// ---------------------------------------------------------------------------
// 配置
// ---------------------------------------------------------------------------

type Config struct {
	Input  string
	Output string
	OutDir string

	Net      string
	Scale    int
	TileSize int
	GPUID    int

	Pages       string
	Redo        bool
	Offline     bool
	ClearCache  bool
	CleanAssets bool
	List        bool
	Open        bool
	Compare     bool
	NoHTML      bool
	NoOrig      bool
	TUI         bool
	Example     bool
	CacheDir    string
	JPEGQuality int
}

func parseFlags() Config {
	var cfg Config
	flag.Usage = usage

	flag.StringVar(&cfg.Input, "i", "", "输入文件：.cbz 或 .zip")
	flag.StringVar(&cfg.Output, "o", "", "输出压缩包：.zip 或 .cbz")
	flag.StringVar(&cfg.OutDir, "outdir", "", "预览目录（提取原图 + 放大图 + HTML 画廊）")

	flag.StringVar(&cfg.Net, "net", "realesrgan-x4plus-anime",
		"模型：realesrgan-x4plus（通用）/ realesrgan-x4plus-anime（漫画推荐）/ realesr-animevideov3")
	flag.IntVar(&cfg.Scale, "scale", 4, "放大倍数（依模型而定：2/3/4）")
	flag.IntVar(&cfg.TileSize, "tile", 0, "ncnn 分块大小（0 = 自动）")
	flag.IntVar(&cfg.GPUID, "gpu", 0, "Vulkan 设备 ID（-1 = CPU，0 = 第一块 GPU）")
	flag.IntVar(&cfg.JPEGQuality, "jpeg-quality", 95, "JPEG 输出质量")

	flag.StringVar(&cfg.Pages, "pages", "", "页选择，例如 \"1,5,10-20\"（默认为全部）")
	flag.BoolVar(&cfg.Redo, "redo", false, "对选中页忽略缓存，重新放大")
	flag.BoolVar(&cfg.Offline, "offline", false, "仅使用缓存，绝不调用模型")
	flag.BoolVar(&cfg.ClearCache, "clear-cache", false, "删除指定书的页缓存目录并退出")
	flag.BoolVar(&cfg.CleanAssets, "clean-assets", false,
		"删除内嵌资源释放目录（下次运行会重新释放），并退出")
	flag.BoolVar(&cfg.List, "list", false, "列出已缓存的页并退出")
	flag.BoolVar(&cfg.Open, "open", false, "完成后打开输出（浏览器/文件管理器）")
	flag.BoolVar(&cfg.Compare, "compare", false, "生成原图/放大图并排对比图")
	flag.BoolVar(&cfg.NoHTML, "no-html", false, "不生成 HTML 画廊")
	flag.BoolVar(&cfg.NoOrig, "no-orig", false, "不提取原始图片")
	flag.BoolVar(&cfg.TUI, "tui", false, "完成后启动终端 TUI 浏览器（需要 -outdir）")
	flag.BoolVar(&cfg.Example, "example", false,
		"打开示例命令 TUI 速查（可模糊过滤，Enter 复制到剪贴板）")
	flag.StringVar(&cfg.CacheDir, "cache-dir", "", "页缓存目录（默认：<输入>.pages）")

	flag.Parse()

	// 纯展示类命令不需要输入文件
	if cfg.Example {
		return cfg
	}
	if cfg.CleanAssets && cfg.Input == "" {
		return cfg
	}

	if cfg.Input == "" {
		usage()
		os.Exit(2)
	}
	needOutput := !cfg.ClearCache && !cfg.List
	if needOutput && cfg.Output == "" && cfg.OutDir == "" {
		fmt.Fprintln(os.Stderr, "错误：必须指定 -o 或 -outdir 之一")
		os.Exit(2)
	}
	if cfg.Output != "" && cfg.OutDir != "" {
		fmt.Fprintln(os.Stderr, "错误：-o 与 -outdir 不能同时使用")
		os.Exit(2)
	}
	if cfg.Compare && cfg.OutDir == "" {
		fmt.Fprintln(os.Stderr, "错误：-compare 需要配合 -outdir 使用")
		os.Exit(2)
	}
	if cfg.TUI && cfg.OutDir == "" {
		fmt.Fprintln(os.Stderr, "错误：-tui 需要配合 -outdir 使用")
		os.Exit(2)
	}
	return cfg
}

func usage() {
	fmt.Fprint(os.Stderr, `upscalecli — 使用 Real-ESRGAN (ncnn/Vulkan) 对 .cbz / .zip 漫画做超分辨率放大。

所有资源（realesrgan-ncnn-vulkan 可执行文件 + 模型）已内嵌在二进制中，
运行时自动释放到系统缓存目录，无需任何外部依赖，一个可执行文件拷到哪都能跑。

内置资源缓存位置：
  Linux   ~/.cache/upscalecli/`+assetsVersion+`/
  macOS   ~/Library/Caches/upscalecli/`+assetsVersion+`/
  Windows %LocalAppData%\upscalecli\`+assetsVersion+`\

──────────── 可直接复制的使用示例 ────────────

  # 0. 打开示例速查 TUI（模糊过滤，Enter 复制到剪贴板）
  upscalecli --example

  # 1. 预览前 6 页，带 HTML 画廊
  upscalecli -i case1.zip -outdir preview/ --pages 1-6 --open

  # 2. 预览前 6 页，完成后直接进终端 TUI 浏览
  upscalecli -i case1.zip -outdir preview/ --pages 1-6 --tui

  # 3. 效果满意，处理整本并输出压缩包
  upscalecli -i case1.zip -o case1.4x.zip

  # 4. 中断了？重新执行相同命令，已缓存的页秒回
  upscalecli -i case1.zip -o case1.4x.zip

  # 5. 想重做第 42 页（比如换了模型）
  upscalecli -i case1.zip -outdir preview/ --pages 42 --redo -net realesrgan-x4plus

  # 6. 无 GPU 的服务器，强制走 CPU（慢，但一定能跑）
  upscalecli -i case1.zip -o case1.4x.zip -gpu -1

  # 7. 生成原图 / 放大图并排对比（用于发帖/分享）
  upscalecli -i case1.zip -outdir review/ --compare

  # 8. 离线重打包（只读缓存，不跑模型）
  upscalecli -i case1.zip -o case1.4x.zip --offline

  # 9. 查看这本书已经缓存了多少页
  upscalecli -i case1.zip -outdir preview/ --list

  # 10. 清除某本书的页缓存
  upscalecli -i case1.zip -outdir preview/ --clear-cache

  # 11. 清除内嵌资源释放目录（下次运行自动重新释放）
  upscalecli --clean-assets

关于元数据自动修正：
  当源压缩包里的图片是 WebP（Go 没有纯 Go 的 WebP 编码器）时，输出会改成 JPEG，
  文件名后缀也会随之变化。为了让 comic_info.json / processed_comic_info.json
  等元数据仍然有效，程序会自动把其中的 .webp 引用替换成 .jpg。

参数说明：
`)
	flag.PrintDefaults()
}

// ---------------------------------------------------------------------------
// Zip 条目
// ---------------------------------------------------------------------------

type entry struct {
	f     *zip.File
	index int // 1 起算的图片序号；非图片条目为 0
}

// ---------------------------------------------------------------------------
// 主流程
// ---------------------------------------------------------------------------

func main() {
	log.SetFlags(log.Ltime)
	cfg := parseFlags()

	// 示例 TUI：不进入正常处理流程
	if cfg.Example {
		runExampleBrowser()
		return
	}

	// 清理内嵌资源释放目录
	if cfg.CleanAssets {
		if err := cleanAssetsDir(); err != nil {
			log.Fatalf("error: %v", err)
		}
		return
	}

	if err := run(cfg); err != nil {
		log.Fatalf("error: %v", err)
	}
}

func run(cfg Config) error {
	if cfg.Output != "" {
		if abs, err := filepath.Abs(cfg.Input); err == nil {
			if outAbs, err := filepath.Abs(cfg.Output); err == nil && abs == outAbs {
				return errors.New("输出路径不能与输入相同")
			}
		}
	}

	cacheDir := cfg.CacheDir
	if cacheDir == "" {
		base := strings.TrimSuffix(cfg.Input, filepath.Ext(cfg.Input))
		cacheDir = base + ".pages"
	}

	if cfg.ClearCache {
		if err := os.RemoveAll(cacheDir); err != nil {
			return fmt.Errorf("清除缓存: %w", err)
		}
		log.Printf("已清除缓存: %s", cacheDir)
		return nil
	}

	r, err := zip.OpenReader(cfg.Input)
	if err != nil {
		return fmt.Errorf("打开输入: %w", err)
	}
	defer r.Close()

	var entries []entry
	totalImages := 0
	for _, f := range r.File {
		if !f.FileInfo().IsDir() && isImageEntry(f.Name) {
			totalImages++
			entries = append(entries, entry{f: f, index: totalImages})
		} else {
			entries = append(entries, entry{f: f, index: 0})
		}
	}
	if totalImages == 0 {
		return errors.New("压缩包中没有找到图片")
	}

	cache, err := openCache(cacheDir, cfg.Net, cfg.Scale, cfg.TileSize)
	if err != nil {
		return fmt.Errorf("打开缓存: %w", err)
	}
	if cfg.List {
		return listCache(entries, totalImages, cache)
	}

	log.Printf("共发现 %d 张图片", totalImages)

	selection, err := parsePages(cfg.Pages, totalImages)
	if err != nil {
		return err
	}
	if cfg.Pages != "" {
		n := 0
		for _, s := range selection {
			if s {
				n++
			}
		}
		log.Printf("选择: 匹配 %d 页 (--pages %q)", n, cfg.Pages)
	}
	if n := len(cache.m.Pages); n > 0 {
		log.Printf("缓存: 已有 %d 页完成", n)
	}

	needRuntime := !cfg.Offline && anySelected(selection)
	var up *VulkanUpscaler
	if needRuntime {
		up, err = NewVulkanUpscaler(cfg.Net, cfg.Scale, cfg.TileSize, cfg.GPUID)
		if err != nil {
			return fmt.Errorf("准备放大器: %w", err)
		}
		defer up.Destroy()
		log.Printf("已就绪 (net=%s scale=%d tile=%d gpu=%d)",
			cfg.Net, cfg.Scale, cfg.TileSize, cfg.GPUID)
	}

	var upscaled, reused, passed int
	var pages []previewPage

	if cfg.OutDir != "" {
		upscaled, reused, passed, pages, err = runPreview(cfg, entries, totalImages, selection, cache, up)
	} else {
		upscaled, reused, passed, err = runArchive(cfg, entries, totalImages, selection, cache, up)
	}
	if err != nil {
		return err
	}

	log.Printf("汇总: 放大 %d 张，复用 %d 张，跳过 %d 张", upscaled, reused, passed)

	// 终端 TUI 浏览（仅预览模式支持）
	if cfg.TUI && cfg.OutDir != "" {
		if err := runTUI(cfg, pages); err != nil {
			log.Printf("TUI 退出: %v", err)
		}
	}

	if cfg.Open {
		target := cfg.Output
		if cfg.OutDir != "" {
			target = filepath.Join(cfg.OutDir, "index.html")
			if cfg.NoHTML {
				target = cfg.OutDir
			}
		}
		if err := openPath(target); err != nil {
			log.Printf("打开输出: %v", err)
		}
	}
	return nil
}

func anySelected(sel []bool) bool {
	for _, s := range sel {
		if s {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// 内嵌资源释放 / 清理
// ---------------------------------------------------------------------------

func assetCacheDir() (string, error) {
	root, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("定位系统缓存目录: %w", err)
	}
	return filepath.Join(root, "upscalecli", assetsVersion), nil
}

// cleanAssetsDir 删除内嵌资源释放目录。下次运行时会被自动重新创建。
func cleanAssetsDir() error {
	root, err := assetCacheDir()
	if err != nil {
		return err
	}
	if err := os.RemoveAll(root); err != nil {
		return fmt.Errorf("删除资源目录: %w", err)
	}
	fmt.Printf("已清除内嵌资源目录: %s\n", root)
	fmt.Println("下次运行时将自动重新释放。")
	return nil
}

// writeFileAtomic 先写临时文件再 rename，避免崩溃留下半个文件。
func writeFileAtomic(path string, data []byte, mode os.FileMode) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, mode); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// ensureAssets 把内嵌的二进制和模型释放到用户缓存目录，只做一次。
func ensureAssets() (binPath, modelsDir string, err error) {
	root, err := assetCacheDir()
	if err != nil {
		return "", "", err
	}

	binPath = filepath.Join(root, binName)
	modelsDir = filepath.Join(root, "models")

	if info, statErr := os.Stat(binPath); statErr == nil && info.Size() == int64(len(embeddedBin)) {
		if _, statErr := os.Stat(modelsDir); statErr == nil {
			return binPath, modelsDir, nil
		}
	}

	log.Printf("首次运行，释放内嵌资源到 %s", root)

	if err := os.MkdirAll(root, 0o755); err != nil {
		return "", "", fmt.Errorf("创建缓存目录: %w", err)
	}
	if err := writeFileAtomic(binPath, embeddedBin, 0o755); err != nil {
		return "", "", fmt.Errorf("释放可执行文件: %w", err)
	}

	if err := os.MkdirAll(modelsDir, 0o755); err != nil {
		return "", "", fmt.Errorf("创建模型目录: %w", err)
	}
	entries, err := embeddedModels.ReadDir("assets/models")
	if err != nil {
		return "", "", fmt.Errorf("读取内嵌模型: %w", err)
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		data, err := embeddedModels.ReadFile("assets/models/" + e.Name())
		if err != nil {
			return "", "", fmt.Errorf("读取模型 %s: %w", e.Name(), err)
		}
		dst := filepath.Join(modelsDir, e.Name())
		if err := writeFileAtomic(dst, data, 0o644); err != nil {
			return "", "", fmt.Errorf("释放模型 %s: %w", e.Name(), err)
		}
	}

	log.Printf("释放完成: %s", binPath)
	return binPath, modelsDir, nil
}

// ---------------------------------------------------------------------------
// 文件重命名映射 —— 元数据里的旧引用需要同步更新
// ---------------------------------------------------------------------------

// outputExtFor 计算一个源扩展名对应的输出扩展名。
// WebP 没有纯 Go 编码器，输出为 JPEG；其他扩展名原样保留。
func outputExtFor(origExt string) string {
	if strings.EqualFold(origExt, ".webp") {
		return ".jpg"
	}
	return origExt
}

// predictOutputName 预测一个图片条目放大后会用什么文件名。
// 只做后缀推导，不做实际编码。
func predictOutputName(origName string) string {
	ext := filepath.Ext(origName)
	newExt := outputExtFor(ext)
	if newExt == ext {
		return origName
	}
	return strings.TrimSuffix(origName, ext) + newExt
}

// isTextMetadata 判断一个条目是不是可能引用图片文件名的文本类元数据。
func isTextMetadata(name string) bool {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".json", ".xml", ".txt", ".opf", ".comicinfo",
		".yaml", ".yml", ".toml", ".csv", ".md", ".html", ".htm":
		return true
	}
	return false
}

// isWordByte 判断一个字节是不是“标识符字符”，用于区分 .webp 到底是
// 文件扩展名还是更长字符串的一部分。
func isWordByte(b byte) bool {
	switch {
	case b >= 'a' && b <= 'z':
		return true
	case b >= 'A' && b <= 'Z':
		return true
	case b >= '0' && b <= '9':
		return true
	case b == '_', b == '-':
		return true
	}
	return false
}

// replaceExtSmart 把内容里的 oldExt 替换为 newExt，但跳过“词内匹配”。
// 例如 ".webp" 后面跟着字母/数字/-/_ 时说明它是更长标识符的一部分，不替换。
func replaceExtSmart(content []byte, oldExt, newExt string) []byte {
	oldB := []byte(oldExt)
	newB := []byte(newExt)
	var out []byte
	i := 0
	for i < len(content) {
		idx := bytes.Index(content[i:], oldB)
		if idx < 0 {
			out = append(out, content[i:]...)
			break
		}
		absIdx := i + idx
		endPos := absIdx + len(oldB)
		// 检查后缀是不是词边界
		if endPos < len(content) && isWordByte(content[endPos]) {
			out = append(out, content[i:endPos]...)
			i = endPos
			continue
		}
		out = append(out, content[i:absIdx]...)
		out = append(out, newB...)
		i = endPos
	}
	return out
}

// applyRenames 依次应用完整路径替换和后缀替换。
// 先做完整路径（最具体、最安全），再做后缀级别（用于只出现基名的情况）。
func applyRenames(content []byte, renameMap, extChanges map[string]string) []byte {
	for old, new := range renameMap {
		content = bytes.ReplaceAll(content, []byte(old), []byte(new))
	}
	for oldExt, newExt := range extChanges {
		content = replaceExtSmart(content, oldExt, newExt)
	}
	return content
}

// buildRenameMaps 预扫描所有图片条目，构建：
//   - renameMap: 完整路径替换表
//   - extChanges: 扩展名级替换表
func buildRenameMaps(entries []entry) (renameMap, extChanges map[string]string, conflicts []string) {
	renameMap = map[string]string{}
	extChanges = map[string]string{}
	seen := map[string]string{} // 最终输出名 → 源条目名，用于冲突检测

	for _, e := range entries {
		if e.index == 0 {
			continue
		}
		oldName := e.f.Name
		newName := predictOutputName(oldName)

		if prev, ok := seen[newName]; ok && prev != oldName {
			conflicts = append(conflicts, fmt.Sprintf("%q ← %q 与 %q", newName, prev, oldName))
		}
		seen[newName] = oldName

		if newName == oldName {
			continue
		}
		renameMap[oldName] = newName
		oldExt := filepath.Ext(oldName)
		newExt := filepath.Ext(newName)
		if oldExt != newExt {
			extChanges[oldExt] = newExt
		}
	}
	return
}

// ---------------------------------------------------------------------------
// Vulkan 放大器 —— 调用内嵌的 realesrgan-ncnn-vulkan 子进程
// ---------------------------------------------------------------------------

type VulkanUpscaler struct {
	binPath   string
	modelsDir string
	net       string
	scale     int
	tileSize  int
	gpuID     int
	tmpDir    string
}

func NewVulkanUpscaler(net string, scale, tileSize, gpuID int) (*VulkanUpscaler, error) {
	binPath, modelsDir, err := ensureAssets()
	if err != nil {
		return nil, err
	}

	paramPath := filepath.Join(modelsDir, net+".param")
	binModelPath := filepath.Join(modelsDir, net+".bin")
	if _, err := os.Stat(paramPath); err != nil {
		return nil, fmt.Errorf("模型 %q 不存在：缺少 %s（请把它放进 assets/models/ 后重新编译）", net, paramPath)
	}
	if _, err := os.Stat(binModelPath); err != nil {
		return nil, fmt.Errorf("模型 %q 不存在：缺少 %s（请把它放进 assets/models/ 后重新编译）", net, binModelPath)
	}

	tmpDir, err := os.MkdirTemp("", "upscalecli-*")
	if err != nil {
		return nil, fmt.Errorf("创建临时目录: %w", err)
	}

	return &VulkanUpscaler{
		binPath:   binPath,
		modelsDir: modelsDir,
		net:       net,
		scale:     scale,
		tileSize:  tileSize,
		gpuID:     gpuID,
		tmpDir:    tmpDir,
	}, nil
}

func (u *VulkanUpscaler) Destroy() {
	if u.tmpDir != "" {
		os.RemoveAll(u.tmpDir)
	}
}

// Upscale 把一张图片交给 realesrgan-ncnn-vulkan 处理。
// realesrgan-ncnn-vulkan 只接受文件路径，所以先落盘为 PNG，处理完再读回来。
func (u *VulkanUpscaler) Upscale(img image.Image) (image.Image, error) {
	inPath := filepath.Join(u.tmpDir, "in.png")
	outPath := filepath.Join(u.tmpDir, "out.png")

	if err := writePNG(inPath, img); err != nil {
		return nil, fmt.Errorf("写入输入: %w", err)
	}

	args := []string{
		"-i", inPath,
		"-o", outPath,
		"-n", u.net,
		"-s", strconv.Itoa(u.scale),
		"-m", u.modelsDir,
		"-g", strconv.Itoa(u.gpuID),
	}
	if u.tileSize > 0 {
		args = append(args, "-t", strconv.Itoa(u.tileSize))
	}

	cmd := exec.Command(u.binPath, args...)
	cmd.Dir = u.tmpDir
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		tail := lastLines(stderr.String(), 5)
		return nil, fmt.Errorf("realesrgan 执行失败: %v\n%s", err, tail)
	}

	out, err := readImage(outPath)
	if err != nil {
		return nil, fmt.Errorf("读取输出: %w", err)
	}
	return out, nil
}

func writePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func readImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

func lastLines(s string, n int) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) <= n {
		return strings.Join(lines, "\n")
	}
	return strings.Join(lines[len(lines)-n:], "\n")
}

// ---------------------------------------------------------------------------
// 压缩包模式 —— 输出一个新的 .cbz / .zip
// ---------------------------------------------------------------------------

func runArchive(cfg Config, entries []entry, total int, selection []bool, cache *PageCache, up *VulkanUpscaler) (int, int, int, error) {
	// 预扫描：找出会被改名的图片条目，并构建元数据替换表
	renameMap, extChanges, conflicts := buildRenameMaps(entries)
	if len(renameMap) > 0 {
		log.Printf("检测到 %d 个条目会改名（例如 WebP → JPEG），将同步更新元数据引用",
			len(renameMap))
	}
	for _, c := range conflicts {
		log.Printf("警告：输出文件名冲突 — %s", c)
	}

	out, err := os.Create(cfg.Output)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("创建输出: %w", err)
	}
	defer out.Close()

	zw := zip.NewWriter(out)
	defer zw.Close()

	var upscaled, reused, passed int
	metaUpdated := 0

	writeEntry := func(name string, mode os.FileMode, modified time.Time, data []byte) error {
		hdr := &zip.FileHeader{Name: name, Method: zip.Deflate, Modified: modified}
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

	for _, e := range entries {
		f := e.f

		if f.FileInfo().IsDir() {
			if err := writeEntry(f.Name, f.Mode(), f.Modified, nil); err != nil {
				return 0, 0, 0, err
			}
			continue
		}

		raw, err := readZipEntry(f)
		if err != nil {
			return 0, 0, 0, fmt.Errorf("读取 %s: %w", f.Name, err)
		}

		if e.index == 0 {
			// 非图片条目：如果是文本类元数据，同步更新里面的图片引用
			if (len(renameMap) > 0 || len(extChanges) > 0) && isTextMetadata(f.Name) {
				rewritten := applyRenames(raw, renameMap, extChanges)
				if !bytes.Equal(rewritten, raw) {
					log.Printf("%s: 更新元数据中的图片引用 (%d → %d 字节)",
						f.Name, len(raw), len(rewritten))
					raw = rewritten
					metaUpdated++
				}
			}
			if err := writeEntry(f.Name, f.Mode(), f.Modified, raw); err != nil {
				return 0, 0, 0, err
			}
			continue
		}

		prefix := fmt.Sprintf("[%d/%d] %s", e.index, total, f.Name)
		useCache := !(cfg.Redo && selection[e.index])

		if useCache {
			if data, outName, ok := cache.get(e.index); ok {
				log.Printf("%s: 命中缓存", prefix)
				if err := writeEntry(outName, f.Mode(), f.Modified, data); err != nil {
					return 0, 0, 0, err
				}
				reused++
				continue
			}
		}
		if !selection[e.index] {
			if err := writeEntry(f.Name, f.Mode(), f.Modified, raw); err != nil {
				return 0, 0, 0, err
			}
			passed++
			continue
		}
		if cfg.Offline || up == nil {
			log.Printf("%s: 无缓存，--offline 或未加载模型，原样复制", prefix)
			if err := writeEntry(f.Name, f.Mode(), f.Modified, raw); err != nil {
				return 0, 0, 0, err
			}
			passed++
			continue
		}

		log.Printf("%s: 放大中…", prefix)
		start := time.Now()

		img, err := decodeImage(raw)
		if err != nil {
			log.Printf("%s: 解码失败 (%v)，原样复制", prefix, err)
			if err := writeEntry(f.Name, f.Mode(), f.Modified, raw); err != nil {
				return 0, 0, 0, err
			}
			passed++
			continue
		}

		result, err := up.Upscale(img)
		if err != nil {
			log.Printf("%s: 放大失败 (%v)，原样复制", prefix, err)
			if err := writeEntry(f.Name, f.Mode(), f.Modified, raw); err != nil {
				return 0, 0, 0, err
			}
			passed++
			continue
		}

		outName, data, err := encodeOutput(f.Name, result, cfg)
		if err != nil {
			return 0, 0, 0, fmt.Errorf("编码 %s: %w", f.Name, err)
		}
		if err := cache.put(e.index, f.Name, outName, data); err != nil {
			log.Printf("%s: 缓存写入失败: %v", prefix, err)
		}
		if err := writeEntry(outName, f.Mode(), f.Modified, data); err != nil {
			return 0, 0, 0, err
		}
		log.Printf("%s: 完成，耗时 %.1fs", prefix, time.Since(start).Seconds())
		upscaled++
	}

	if metaUpdated > 0 {
		log.Printf("已更新 %d 个元数据文件中的图片引用", metaUpdated)
	}

	return upscaled, reused, passed, nil
}

// ---------------------------------------------------------------------------
// 预览模式 —— 输出到目录，便于人工检查
// ---------------------------------------------------------------------------

type previewPage struct {
	Index   int    // 1 起算的页号
	Name    string // 原始 zip 内的路径
	OrigRel string // 相对 outdir 的原图路径（正斜杠）
	UpRel   string // 相对 outdir 的放大图路径（正斜杠）
}

func runPreview(cfg Config, entries []entry, total int, selection []bool, cache *PageCache, up *VulkanUpscaler) (int, int, int, []previewPage, error) {
	origDir := filepath.Join(cfg.OutDir, dirOriginal)
	upDir := filepath.Join(cfg.OutDir, dirUpscaled)
	cmpDir := filepath.Join(cfg.OutDir, dirCompare)

	if !cfg.NoOrig {
		if err := os.MkdirAll(origDir, 0o755); err != nil {
			return 0, 0, 0, nil, err
		}
	}
	if err := os.MkdirAll(upDir, 0o755); err != nil {
		return 0, 0, 0, nil, err
	}
	if cfg.Compare {
		if err := os.MkdirAll(cmpDir, 0o755); err != nil {
			return 0, 0, 0, nil, err
		}
	}

	var pages []previewPage
	var upscaled, reused, passed int

	writeFile := func(root, rel string, data []byte) error {
		full := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			return err
		}
		return os.WriteFile(full, data, 0o644)
	}

	for _, e := range entries {
		f := e.f

		if f.FileInfo().IsDir() {
			continue
		}

		raw, err := readZipEntry(f)
		if err != nil {
			return 0, 0, 0, nil, fmt.Errorf("读取 %s: %w", f.Name, err)
		}

		if e.index == 0 {
			if !cfg.NoOrig {
				if err := writeFile(origDir, f.Name, raw); err != nil {
					return 0, 0, 0, nil, err
				}
			}
			continue
		}

		prefix := fmt.Sprintf("[%d/%d] %s", e.index, total, f.Name)

		// 原图总是先落地，方便对比
		if !cfg.NoOrig {
			if err := writeFile(origDir, f.Name, raw); err != nil {
				return 0, 0, 0, nil, err
			}
		}

		useCache := !(cfg.Redo && selection[e.index])

		var outName string
		var outData []byte

		if useCache {
			if data, name, ok := cache.get(e.index); ok {
				log.Printf("%s: 命中缓存", prefix)
				outName, outData = name, data
				reused++
			}
		}

		if outData == nil {
			if !selection[e.index] {
				outName, outData = f.Name, raw
				passed++
			} else if cfg.Offline || up == nil {
				log.Printf("%s: 无缓存，原样复制", prefix)
				outName, outData = f.Name, raw
				passed++
			} else {
				log.Printf("%s: 放大中…", prefix)
				start := time.Now()

				img, err := decodeImage(raw)
				if err != nil {
					log.Printf("%s: 解码失败 (%v)", prefix, err)
					outName, outData = f.Name, raw
					passed++
				} else {
					result, err := up.Upscale(img)
					if err != nil {
						log.Printf("%s: 放大失败 (%v)", prefix, err)
						outName, outData = f.Name, raw
						passed++
					} else {
						name, data, err := encodeOutput(f.Name, result, cfg)
						if err != nil {
							return 0, 0, 0, nil, fmt.Errorf("编码 %s: %w", f.Name, err)
						}
						if err := cache.put(e.index, f.Name, name, data); err != nil {
							log.Printf("%s: 缓存写入失败: %v", prefix, err)
						}
						outName, outData = name, data
						log.Printf("%s: 完成，耗时 %.1fs", prefix, time.Since(start).Seconds())
						upscaled++
					}
				}
			}
		}

		if err := writeFile(upDir, outName, outData); err != nil {
			return 0, 0, 0, nil, err
		}

		// 可选：生成并排对比图
		if cfg.Compare {
			composite, err := makeCompare(raw, outData)
			if err != nil {
				log.Printf("%s: 对比图生成失败: %v", prefix, err)
			} else {
				base := strings.TrimSuffix(filepath.Base(outName), filepath.Ext(outName)) + ".jpg"
				rel := filepath.ToSlash(filepath.Join(filepath.Dir(outName), base))
				if err := writeFile(cmpDir, rel, composite); err != nil {
					return 0, 0, 0, nil, err
				}
			}
		}

		pages = append(pages, previewPage{
			Index:   e.index,
			Name:    f.Name,
			OrigRel: filepath.ToSlash(f.Name),
			UpRel:   filepath.ToSlash(outName),
		})
	}

	// HTML 画廊（可选）
	if !cfg.NoHTML {
		if err := writeGallery(cfg, pages); err != nil {
			return 0, 0, 0, nil, fmt.Errorf("生成 HTML: %w", err)
		}
	}
	return upscaled, reused, passed, pages, nil
}

// ---------------------------------------------------------------------------
// 并排对比图
// ---------------------------------------------------------------------------

func makeCompare(origBytes, upBytes []byte) ([]byte, error) {
	orig, _, err := image.Decode(bytes.NewReader(origBytes))
	if err != nil {
		return nil, err
	}
	up, _, err := image.Decode(bytes.NewReader(upBytes))
	if err != nil {
		return nil, err
	}

	ob, ub := orig.Bounds(), up.Bounds()
	origH := ob.Dy()
	targetH := ub.Dy()
	if origH != targetH {
		ratio := float64(targetH) / float64(origH)
		newW := int(float64(ob.Dx()) * ratio)
		scaled := image.NewRGBA(image.Rect(0, 0, newW, targetH))
		xdraw.CatmullRom.Scale(scaled, scaled.Bounds(), orig, ob, xdraw.Over, nil)
		orig = scaled
		ob = orig.Bounds()
	}

	const gap = 20
	W := ob.Dx() + gap + ub.Dx()
	H := targetH
	dst := image.NewRGBA(image.Rect(0, 0, W, H))
	draw.Draw(dst, dst.Bounds(), image.NewUniform(color.RGBA{25, 25, 25, 255}), image.Point{}, draw.Src)
	draw.Draw(dst, image.Rect(0, 0, ob.Dx(), ob.Dy()), orig, ob.Min, draw.Src)
	draw.Draw(dst, image.Rect(ob.Dx()+gap, 0, ob.Dx()+gap+ub.Dx(), ub.Dy()), up, ub.Min, draw.Src)

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, dst, &jpeg.Options{Quality: 85}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// ---------------------------------------------------------------------------
// 剪贴板 —— 支持 Wayland / X11 / macOS / Windows
// ---------------------------------------------------------------------------

func clipboardCommand() (*exec.Cmd, error) {
	if os.Getenv("WAYLAND_DISPLAY") != "" {
		if p, err := exec.LookPath("wl-copy"); err == nil {
			return exec.Command(p), nil
		}
	}
	if os.Getenv("DISPLAY") != "" {
		if p, err := exec.LookPath("xclip"); err == nil {
			return exec.Command(p, "-selection", "clipboard"), nil
		}
		if p, err := exec.LookPath("xsel"); err == nil {
			return exec.Command(p, "--clipboard", "--input"), nil
		}
	}
	switch runtime.GOOS {
	case "darwin":
		if p, err := exec.LookPath("pbcopy"); err == nil {
			return exec.Command(p), nil
		}
	case "windows":
		if p, err := exec.LookPath("clip"); err == nil {
			return exec.Command(p), nil
		}
	}
	return nil, errors.New(
		"未找到剪贴板工具：\n" +
			"  Wayland 请安装 wl-clipboard（提供 wl-copy）\n" +
			"  X11     请安装 xclip 或 xsel")
}

func copyToClipboard(text string) error {
	cmd, err := clipboardCommand()
	if err != nil {
		return err
	}
	cmd.Stdin = strings.NewReader(text)
	return cmd.Run()
}

// ---------------------------------------------------------------------------
// 示例速查 TUI（模糊过滤 + Enter 复制）
// ---------------------------------------------------------------------------

type exampleItem struct {
	title string
	cmd   string
}

var builtinExamples = []exampleItem{
	{"预览前 6 页（带 HTML 画廊）",
		"upscalecli -i case1.zip -outdir preview/ --pages 1-6 --open"},
	{"预览前 6 页（完成后进终端 TUI 浏览）",
		"upscalecli -i case1.zip -outdir preview/ --pages 1-6 --tui"},
	{"处理整本并输出压缩包",
		"upscalecli -i case1.zip -o case1.4x.zip"},
	{"断点续传：中断后重新执行相同命令即可",
		"upscalecli -i case1.zip -o case1.4x.zip"},
	{"重做第 42 页（例如换了模型之后）",
		"upscalecli -i case1.zip -outdir preview/ --pages 42 --redo -net realesrgan-x4plus"},
	{"无 GPU 服务器：强制走 CPU",
		"upscalecli -i case1.zip -o case1.4x.zip -gpu -1"},
	{"生成原图 / 放大图并排对比",
		"upscalecli -i case1.zip -outdir review/ --compare"},
	{"离线重打包（只读缓存，不跑模型）",
		"upscalecli -i case1.zip -o case1.4x.zip --offline"},
	{"查看这本书已经缓存了多少页",
		"upscalecli -i case1.zip -outdir preview/ --list"},
	{"清除某本书的页缓存",
		"upscalecli -i case1.zip -outdir preview/ --clear-cache"},
	{"清除内嵌资源释放目录（下次运行自动重建）",
		"upscalecli --clean-assets"},
	{"换用通用模型（照片 / 写实画面）",
		"upscalecli -i case1.zip -o case1.4x.zip -net realesrgan-x4plus"},
	{"换用动画视频帧模型，2 倍放大",
		"upscalecli -i case1.zip -o case1.2x.zip -net realesr-animevideov3 -scale 2"},
	{"限制 tile 大小以节省显存",
		"upscalecli -i case1.zip -o case1.4x.zip -tile 128"},
	{"自定义 JPEG 输出质量",
		"upscalecli -i case1.zip -o case1.4x.zip -jpeg-quality 88"},
	{"指定页缓存目录（避免放在只读输入旁边）",
		"upscalecli -i case1.zip -o case1.4x.zip --cache-dir /tmp/case1.pages"},
}

type exampleModel struct {
	items    []exampleItem
	filtered []int
	query    string
	cursor   int
	width    int
	height   int
	status   string
	chosen   *exampleItem
}

func newExampleModel() exampleModel {
	m := exampleModel{
		items:  builtinExamples,
		width:  80,
		height: 24,
	}
	m.applyFilter()
	return m
}

func (m *exampleModel) applyFilter() {
	q := strings.ToLower(m.query)
	m.filtered = m.filtered[:0]
	for i, e := range m.items {
		if q == "" ||
			strings.Contains(strings.ToLower(e.title), q) ||
			strings.Contains(strings.ToLower(e.cmd), q) {
			m.filtered = append(m.filtered, i)
		}
	}
	if m.cursor >= len(m.filtered) {
		m.cursor = len(m.filtered) - 1
	}
	if m.cursor < 0 {
		m.cursor = 0
	}
}

func (m exampleModel) Init() tea.Cmd { return nil }

func (m exampleModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC, tea.KeyEsc:
			return m, tea.Quit
		case tea.KeyEnter:
			if len(m.filtered) == 0 {
				return m, nil
			}
			it := m.items[m.filtered[m.cursor]]
			if err := copyToClipboard(it.cmd); err != nil {
				m.status = "复制失败：" + err.Error()
				return m, nil
			}
			m.chosen = &it
			return m, tea.Quit
		case tea.KeyUp, tea.KeyCtrlP:
			if m.cursor > 0 {
				m.cursor--
			}
		case tea.KeyDown, tea.KeyCtrlN:
			if m.cursor < len(m.filtered)-1 {
				m.cursor++
			}
		case tea.KeyHome:
			m.cursor = 0
		case tea.KeyEnd:
			if len(m.filtered) > 0 {
				m.cursor = len(m.filtered) - 1
			}
		case tea.KeyBackspace:
			if len(m.query) > 0 {
				m.query = m.query[:len(m.query)-1]
				m.applyFilter()
			}
		case tea.KeyDelete:
			m.query = ""
			m.applyFilter()
		case tea.KeyRunes:
			m.query += string(msg.Runes)
			m.applyFilter()
		case tea.KeySpace:
			m.query += " "
			m.applyFilter()
		}
	}
	return m, nil
}

func (m exampleModel) View() string {
	var (
		titleStyle  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("205"))
		labelStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("241"))
		queryStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("39"))
		cursorStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("205"))
		itemStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("252"))
		dimStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
		cmdStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("78"))
		statusStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("203"))
	)

	var sb strings.Builder

	sb.WriteString(titleStyle.Render("upscalecli 示例速查"))
	sb.WriteString("  ")
	sb.WriteString(labelStyle.Render("输入关键字过滤 · Enter 复制到剪贴板并退出"))
	sb.WriteString("\n\n")

	sb.WriteString(labelStyle.Render("过滤: "))
	if m.query == "" {
		sb.WriteString(dimStyle.Render("（直接输入以过滤）"))
	} else {
		sb.WriteString(queryStyle.Render(m.query))
	}
	sb.WriteString(dimStyle.Render("█"))
	sb.WriteString("\n\n")

	listHeight := m.height - 12
	if listHeight < 3 {
		listHeight = 3
	}
	if len(m.filtered) == 0 {
		sb.WriteString(dimStyle.Render("没有匹配的示例。"))
		sb.WriteString("\n")
	} else {
		start := 0
		if m.cursor >= listHeight {
			start = m.cursor - listHeight + 1
		}
		end := start + listHeight
		if end > len(m.filtered) {
			end = len(m.filtered)
		}
		for vi := start; vi < end; vi++ {
			idx := m.filtered[vi]
			it := m.items[idx]
			if vi == m.cursor {
				sb.WriteString(cursorStyle.Render("▸ "))
				sb.WriteString(cursorStyle.Render(it.title))
			} else {
				sb.WriteString(itemStyle.Render("  " + it.title))
			}
			sb.WriteString("\n")
		}
	}

	sb.WriteString("\n")
	sb.WriteString(labelStyle.Render("命令预览:"))
	sb.WriteString("\n")
	if len(m.filtered) > 0 {
		sb.WriteString("  ")
		sb.WriteString(cmdStyle.Render(m.items[m.filtered[m.cursor]].cmd))
	}
	sb.WriteString("\n")

	if m.status != "" {
		sb.WriteString(statusStyle.Render(m.status))
		sb.WriteString("\n")
	}

	sb.WriteString(dimStyle.Render("↑/↓ 选择 · Enter 复制并退出 · Esc 取消 · 输入字符过滤 · Backspace 删除"))
	return sb.String()
}

func runExampleBrowser() {
	if !isTerminal(os.Stdout) {
		for _, e := range builtinExamples {
			fmt.Printf("# %s\n%s\n\n", e.title, e.cmd)
		}
		return
	}

	if _, err := clipboardCommand(); err != nil {
		fmt.Fprintln(os.Stderr, "提示：", err)
		fmt.Fprintln(os.Stderr, "（TUI 仍可使用，但 Enter 无法自动复制）")
	}

	p := tea.NewProgram(newExampleModel(), tea.WithAltScreen())
	finalModel, err := p.Run()
	if err != nil {
		fmt.Fprintln(os.Stderr, "TUI 启动失败:", err)
		for _, e := range builtinExamples {
			fmt.Printf("# %s\n%s\n\n", e.title, e.cmd)
		}
		return
	}

	if m, ok := finalModel.(exampleModel); ok && m.chosen != nil {
		fmt.Println("已复制到剪贴板：")
		fmt.Println(m.chosen.cmd)
	}
}

func isTerminal(f *os.File) bool {
	fi, err := f.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

// ---------------------------------------------------------------------------
// 预览 TUI（bubbletea）—— 用半块字符显示图片
// ---------------------------------------------------------------------------

type tuiModel struct {
	pages    []previewPage
	outDir   string
	index    int
	mode     string
	width    int
	height   int
	imageStr string
}

func initialTUIModel(outDir string, pages []previewPage) tuiModel {
	return tuiModel{
		pages:  pages,
		outDir: outDir,
		index:  0,
		mode:   "upscaled",
		width:  80,
		height: 24,
	}
}

func (m tuiModel) Init() tea.Cmd { return nil }

func (m tuiModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.imageStr = ""
	case tea.KeyMsg:
		switch msg.String() {
		case "q", "esc", "ctrl+c":
			return m, tea.Quit
		case "left", "h", "a":
			if m.index > 0 {
				m.index--
				m.imageStr = ""
			}
		case "right", "l", "d":
			if m.index < len(m.pages)-1 {
				m.index++
				m.imageStr = ""
			}
		case "home", "g":
			m.index = 0
			m.imageStr = ""
		case "end", "G":
			m.index = len(m.pages) - 1
			m.imageStr = ""
		case " ", "tab":
			if m.mode == "orig" {
				m.mode = "upscaled"
			} else {
				m.mode = "orig"
			}
			m.imageStr = ""
		case "o":
			m.mode = "orig"
			m.imageStr = ""
		case "u", "t":
			m.mode = "upscaled"
			m.imageStr = ""
		}
	}
	return m, nil
}

func (m tuiModel) View() string {
	if len(m.pages) == 0 {
		return "没有可预览的页面。按 q 退出。"
	}

	if m.imageStr == "" {
		img, err := m.loadCurrentImage()
		if err != nil {
			m.imageStr = "（无法加载图片： " + err.Error() + "）"
		} else {
			m.imageStr = renderImageHalfBlock(img, m.width, m.height-5)
		}
	}

	p := m.pages[m.index]
	modeLabel := "放大图"
	if m.mode == "orig" {
		modeLabel = "原图"
	}

	titleStyle := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("205"))
	infoStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("241"))
	hintStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("240"))

	bar := titleStyle.Render("Real-ESRGAN 预览") +
		infoStyle.Render(fmt.Sprintf("  %d/%d  ·  %s  ·  %s",
			m.index+1, len(m.pages), p.Name, modeLabel))

	hint := hintStyle.Render(
		"←/→ 翻页 · 空格 切换原图/放大图 · o 原图 · u 放大图 · g/G 首尾 · q 退出")

	return bar + "\n\n" + m.imageStr + "\n" + hint
}

func (m tuiModel) loadCurrentImage() (image.Image, error) {
	p := m.pages[m.index]
	var rel string
	if m.mode == "orig" {
		rel = filepath.Join(dirOriginal, p.OrigRel)
	} else {
		rel = filepath.Join(dirUpscaled, p.UpRel)
	}
	full := filepath.Join(m.outDir, rel)
	return readImage(full)
}

// renderImageHalfBlock 把图像缩放到终端尺寸，再用半块字符（▀）显示。
func renderImageHalfBlock(img image.Image, maxWidth, maxHeight int) string {
	if maxWidth < 4 {
		maxWidth = 4
	}
	if maxHeight < 2 {
		maxHeight = 2
	}

	b := img.Bounds()
	iw, ih := b.Dx(), b.Dy()
	if iw == 0 || ih == 0 {
		return "（空图像）"
	}

	targetW := maxWidth
	targetH := targetW * ih / (iw * 2)
	if targetH > maxHeight {
		targetH = maxHeight
		targetW = targetH * 2 * iw / ih
	}
	if targetW < 1 {
		targetW = 1
	}
	if targetH < 1 {
		targetH = 1
	}

	scaledH := targetH * 2
	scaled := image.NewRGBA(image.Rect(0, 0, targetW, scaledH))
	xdraw.CatmullRom.Scale(scaled, scaled.Bounds(), img, b, xdraw.Over, nil)

	var sb strings.Builder
	for y := 0; y < targetH; y++ {
		for x := 0; x < targetW; x++ {
			topR, topG, topB, _ := scaled.At(x, y*2).RGBA()
			botR, botG, botB, _ := scaled.At(x, y*2+1).RGBA()
			fmt.Fprintf(&sb,
				"\x1b[38;2;%d;%d;%dm\x1b[48;2;%d;%d;%dm▀",
				topR>>8, topG>>8, topB>>8,
				botR>>8, botG>>8, botB>>8,
			)
		}
		sb.WriteString("\x1b[0m\n")
	}
	return sb.String()
}

func runTUI(cfg Config, pages []previewPage) error {
	if len(pages) == 0 {
		return errors.New("没有可预览的页面")
	}
	model := initialTUIModel(cfg.OutDir, pages)
	p := tea.NewProgram(model, tea.WithAltScreen())
	_, err := p.Run()
	return err
}

// ---------------------------------------------------------------------------
// 页缓存
// ---------------------------------------------------------------------------

type PageCache struct {
	dir          string
	manifestPath string
	m            *Manifest
}

type Manifest struct {
	Model    string            `json:"model"`
	Scale    int               `json:"scale"`
	TileSize int               `json:"tile_size"`
	Pages    map[int]PageEntry `json:"pages"`
}

type PageEntry struct {
	Name     string `json:"name"`
	OutName  string `json:"out_name"`
	FileName string `json:"file_name"`
	DoneAt   string `json:"done_at"`
}

func openCache(dir, model string, scale, tile int) (*PageCache, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	c := &PageCache{
		dir:          dir,
		manifestPath: filepath.Join(dir, "manifest.json"),
	}
	data, err := os.ReadFile(c.manifestPath)
	if errors.Is(err, os.ErrNotExist) {
		c.m = &Manifest{Model: model, Scale: scale, TileSize: tile, Pages: map[int]PageEntry{}}
		return c, nil
	}
	if err != nil {
		return nil, err
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("解析 manifest: %w", err)
	}
	if m.Pages == nil {
		m.Pages = map[int]PageEntry{}
	}
	c.m = &m
	if m.Model != model || m.Scale != scale || m.TileSize != tile {
		log.Printf("警告: 缓存使用 model=%q scale=%d tile=%d，"+
			"本次使用 model=%q scale=%d tile=%d",
			m.Model, m.Scale, m.TileSize, model, scale, tile)
		log.Printf("      使用 --redo 忽略缓存页，或 --clear-cache 完全清除")
	}
	return c, nil
}

func (c *PageCache) get(index int) ([]byte, string, bool) {
	meta, ok := c.m.Pages[index]
	if !ok {
		return nil, "", false
	}
	data, err := os.ReadFile(filepath.Join(c.dir, meta.FileName))
	if err != nil {
		return nil, "", false
	}
	return data, meta.OutName, true
}

func (c *PageCache) put(index int, name, outName string, data []byte) error {
	ext := filepath.Ext(outName)
	if ext == "" {
		ext = ".bin"
	}
	fileName := fmt.Sprintf("%04d%s", index, ext)
	if err := os.WriteFile(filepath.Join(c.dir, fileName), data, 0o644); err != nil {
		return err
	}
	c.m.Pages[index] = PageEntry{
		Name:     name,
		OutName:  outName,
		FileName: fileName,
		DoneAt:   time.Now().UTC().Format(time.RFC3339),
	}
	return c.save()
}

func (c *PageCache) save() error {
	data, err := json.MarshalIndent(c.m, "", "  ")
	if err != nil {
		return err
	}
	tmp := c.manifestPath + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, c.manifestPath)
}

// ---------------------------------------------------------------------------
// Zip / 图像辅助
// ---------------------------------------------------------------------------

func readZipEntry(f *zip.File) ([]byte, error) {
	rc, err := f.Open()
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	return io.ReadAll(rc)
}

func decodeImage(raw []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	return img, nil
}

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

// encodeOutput 输出图片。输出文件名与 predictOutputName 保持一致：
// WebP 源会被改名成 .jpg（Go 没有纯 Go 的 WebP 编码器）。
func encodeOutput(origName string, img image.Image, cfg Config) (string, []byte, error) {
	origExt := filepath.Ext(origName)
	outExt := outputExtFor(origExt)
	outName := strings.TrimSuffix(origName, origExt) + outExt

	var buf bytes.Buffer
	switch strings.ToLower(outExt) {
	case ".jpg", ".jpeg":
		if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: cfg.JPEGQuality}); err != nil {
			return "", nil, err
		}
	case ".png":
		if err := png.Encode(&buf, img); err != nil {
			return "", nil, err
		}
	default:
		// 兜底：未知扩展名按 PNG 输出
		if err := png.Encode(&buf, img); err != nil {
			return "", nil, err
		}
	}
	return outName, buf.Bytes(), nil
}

// ---------------------------------------------------------------------------
// 页选择
// ---------------------------------------------------------------------------

func parsePages(spec string, total int) ([]bool, error) {
	sel := make([]bool, total+1)
	if strings.TrimSpace(spec) == "" {
		for i := 1; i <= total; i++ {
			sel[i] = true
		}
		return sel, nil
	}
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if strings.Contains(part, "-") {
			bounds := strings.SplitN(part, "-", 2)
			lo, err := strconv.Atoi(strings.TrimSpace(bounds[0]))
			if err != nil {
				return nil, fmt.Errorf("无效的页范围 %q: %w", part, err)
			}
			hi, err := strconv.Atoi(strings.TrimSpace(bounds[1]))
			if err != nil {
				return nil, fmt.Errorf("无效的页范围 %q: %w", part, err)
			}
			if lo < 1 {
				lo = 1
			}
			if hi > total {
				hi = total
			}
			if lo > hi {
				lo, hi = hi, lo
			}
			for i := lo; i <= hi; i++ {
				sel[i] = true
			}
		} else {
			n, err := strconv.Atoi(part)
			if err != nil {
				return nil, fmt.Errorf("无效的页码 %q: %w", part, err)
			}
			if n >= 1 && n <= total {
				sel[n] = true
			}
		}
	}
	return sel, nil
}

// ---------------------------------------------------------------------------
// 缓存列表
// ---------------------------------------------------------------------------

func listCache(entries []entry, total int, cache *PageCache) error {
	if len(cache.m.Pages) == 0 {
		fmt.Println("缓存为空")
		return nil
	}
	byIndex := map[int]string{}
	for _, e := range entries {
		if e.index > 0 {
			byIndex[e.index] = e.f.Name
		}
	}
	fmt.Printf("已缓存页 (model=%s scale=%d tile=%d):\n",
		cache.m.Model, cache.m.Scale, cache.m.TileSize)
	for i := 1; i <= total; i++ {
		entry, ok := cache.m.Pages[i]
		if !ok {
			continue
		}
		name := byIndex[i]
		if name == "" {
			name = entry.Name
		}
		fmt.Printf("  %4d  %-45s  %s\n", i, truncate(name, 45), entry.DoneAt)
	}
	fmt.Printf("总计: %d / %d 页已缓存\n", len(cache.m.Pages), total)
	return nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// ---------------------------------------------------------------------------
// HTML 画廊（浏览器友好版）
// ---------------------------------------------------------------------------

func writeGallery(cfg Config, pages []previewPage) error {
	type item struct {
		Name string `json:"name"`
		Orig string `json:"orig"`
		Up   string `json:"up"`
	}
	items := make([]item, 0, len(pages))
	for _, p := range pages {
		items = append(items, item{
			Name: p.Name,
			Orig: dirOriginal + "/" + p.OrigRel,
			Up:   dirUpscaled + "/" + p.UpRel,
		})
	}
	payload, err := json.Marshal(items)
	if err != nil {
		return err
	}

	html := strings.ReplaceAll(galleryTemplate, "__PAGES__", string(payload))
	html = strings.ReplaceAll(html, "__MODEL__", htmlEscape(cfg.Net))
	html = strings.ReplaceAll(html, "__COUNT__", strconv.Itoa(len(items)))

	out := filepath.Join(cfg.OutDir, "index.html")
	return os.WriteFile(out, []byte(html), 0o644)
}

func htmlEscape(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	s = strings.ReplaceAll(s, "\"", "&quot;")
	return s
}

const galleryTemplate = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>共 __COUNT__ 页 — __MODEL__</title>
<style>
  html, body { margin: 0; padding: 0; background: #111; color: #ddd;
               font: 14px/1.4 system-ui, -apple-system, sans-serif; }
  #bar { position: fixed; top: 0; left: 0; right: 0; height: 44px;
         background: rgba(20,20,20,0.92); display: flex; align-items: center;
         gap: 10px; padding: 0 12px; z-index: 100; box-shadow: 0 1px 4px #000; }
  #bar button { padding: 6px 12px; background: #2a2a2a; color: #ddd;
                border: 1px solid #444; border-radius: 4px; cursor: pointer; }
  #bar button:hover { background: #383838; }
  #bar .grow { flex: 1; }
  #bar .info { color: #888; font-variant-numeric: tabular-nums; }
  #bar .badge { background: #333; padding: 3px 8px; border-radius: 3px; color: #aaa; }
  #bar .badge.up  { background: #2d4a2d; color: #b8e6b8; }
  #bar .badge.orig { background: #4a3d2d; color: #e6d0a8; }
  #stage { display: flex; align-items: center; justify-content: center;
           min-height: 100vh; padding-top: 60px; box-sizing: border-box; }
  #stage img { max-width: 100vw; max-height: calc(100vh - 80px); display: block; }
  #hint { position: fixed; bottom: 8px; right: 12px; color: #555; font-size: 12px; }
  kbd { background: #2a2a2a; padding: 1px 5px; border-radius: 3px; border: 1px solid #444; }
</style>
</head>
<body>
<div id="bar">
  <button id="btn-toggle">显示原图</button>
  <button id="btn-prev">← 上一页</button>
  <button id="btn-next">下一页 →</button>
  <span class="grow"></span>
  <span class="badge" id="badge">—</span>
  <span class="info" id="info">— / __COUNT__</span>
</div>
<div id="stage"><img id="img" alt=""></div>
<div id="hint">
  <kbd>←</kbd> <kbd>→</kbd> 翻页 · <kbd>空格</kbd> 切换 · <kbd>O</kbd> 原图 · <kbd>T</kbd> 放大图
</div>
<script>
  const pages = __PAGES__;
  let i = 0;
  let mode = "up";
  const $img   = document.getElementById("img");
  const $info  = document.getElementById("info");
  const $badge = document.getElementById("badge");
  const $btnT  = document.getElementById("btn-toggle");

  function render() {
    if (!pages.length) return;
    const p = pages[i];
    $img.src = (mode === "orig") ? p.orig : p.up;
    $info.textContent = (i+1) + " / " + pages.length;
    $badge.textContent = mode === "orig" ? "原图" : "放大图";
    $badge.className = "badge " + mode;
    $btnT.textContent = mode === "orig" ? "显示放大图" : "显示原图";
    document.title = p.name + " — " + (i+1) + "/" + pages.length;
  }
  function setMode(m) { mode = m; render(); }
  function toggle()   { setMode(mode === "up" ? "orig" : "up"); }
  function prev()     { if (i > 0) { i--; render(); } }
  function next()     { if (i < pages.length - 1) { i++; render(); } }

  $btnT.onclick = toggle;
  document.getElementById("btn-prev").onclick = prev;
  document.getElementById("btn-next").onclick = next;
  document.addEventListener("keydown", e => {
    if (e.target.matches("input, textarea")) return;
    switch (e.key) {
      case "ArrowLeft":  prev(); break;
      case "ArrowRight": next(); break;
      case " ":          e.preventDefault(); toggle(); break;
      case "o": case "O": setMode("orig"); break;
      case "t": case "T": setMode("up"); break;
      case "Home": i = 0; render(); break;
      case "End":  i = pages.length - 1; render(); break;
    }
  });
  render();
</script>
</body>
</html>
`

// ---------------------------------------------------------------------------
// 用系统默认程序打开
// ---------------------------------------------------------------------------

func openPath(p string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", p)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", p)
	default:
		cmd = exec.Command("xdg-open", p)
	}
	return cmd.Start()
}
