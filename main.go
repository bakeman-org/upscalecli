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

	xdraw "golang.org/x/image/draw"
	_ "golang.org/x/image/webp"

	_ "image/gif"
)

//go:embed assets/realesrgan-ncnn-vulkan
var embeddedBin []byte

//go:embed assets/models/*
var embeddedModels embed.FS

const (
	assetsVersion = "v1"
	binName       = "realesrgan-ncnn-vulkan"
)

// ---------------------------------------------------------------------------
// Config
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
	List        bool
	Open        bool
	Compare     bool
	NoHTML      bool
	NoOrig      bool
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
	flag.BoolVar(&cfg.ClearCache, "clear-cache", false, "删除缓存目录并退出")
	flag.BoolVar(&cfg.List, "list", false, "列出已缓存的页并退出")
	flag.BoolVar(&cfg.Open, "open", false, "完成后打开输出")
	flag.BoolVar(&cfg.Compare, "compare", false, "生成原图/放大图并排对比图")
	flag.BoolVar(&cfg.NoHTML, "no-html", false, "不生成 HTML 画廊")
	flag.BoolVar(&cfg.NoOrig, "no-orig", false, "不提取原始图片")
	flag.StringVar(&cfg.CacheDir, "cache-dir", "", "缓存目录（默认：<输入>.pages）")

	flag.Parse()

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

使用示例：

  # 1. 预览前 6 页（带 HTML 画廊，自动打开浏览器）
  upscalecli -i case1.zip -outdir preview/ --pages 1-6 --open

  # 2. 看效果满意后，处理整本并输出压缩包
  upscalecli -i case1.zip -o case1.4x.zip

  # 3. 中断了？直接重新执行相同命令，已缓存的页秒回
  upscalecli -i case1.zip -o case1.4x.zip

  # 4. 想重做第 42 页（比如换了模型）
  upscalecli -i case1.zip -outdir preview/ --pages 42 --redo -net realesrgan-x4plus

  # 5. 无 GPU 的服务器，强制走 CPU（慢，但一定能跑）
  upscalecli -i case1.zip -o case1.4x.zip -gpu -1

  # 6. 生成原图/放大图并排对比（用于发帖/分享）
  upscalecli -i case1.zip -outdir review/ --compare

  # 7. 离线重打包（只读缓存，不跑模型）
  upscalecli -i case1.zip -o case1.4x.zip --offline

  # 8. 清空缓存从零开始
  upscalecli -i case1.zip -outdir /tmp/x --clear-cache

参数说明：
`)
	flag.PrintDefaults()
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

type entry struct {
	f     *zip.File
	index int
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	log.SetFlags(log.Ltime)
	cfg := parseFlags()
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

	var translated, reused, passed int

	if cfg.OutDir != "" {
		translated, reused, passed, err = runPreview(cfg, entries, totalImages, selection, cache, up)
	} else {
		translated, reused, passed, err = runArchive(cfg, entries, totalImages, selection, cache, up)
	}
	if err != nil {
		return err
	}

	log.Printf("汇总: 放大 %d 张，复用 %d 张，跳过 %d 张",
		translated, reused, passed)

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
// Asset release — bake realesrgan-ncnn-vulkan + models into the binary
// ---------------------------------------------------------------------------

func assetCacheDir() (string, error) {
	root, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("定位系统缓存目录: %w", err)
	}
	return filepath.Join(root, "upscalecli", assetsVersion), nil
}

func writeFileAtomic(path string, data []byte, mode os.FileMode) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, mode); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// ensureAssets extracts the embedded binary and models to the per-user cache
// directory once, and returns the paths.  Subsequent runs reuse the cache.
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
// Vulkan upscaler — runs the embedded realesrgan-ncnn-vulkan
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
// Archive mode
// ---------------------------------------------------------------------------

func runArchive(cfg Config, entries []entry, total int, selection []bool, cache *PageCache, up *VulkanUpscaler) (int, int, int, error) {
	out, err := os.Create(cfg.Output)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("创建输出: %w", err)
	}
	defer out.Close()

	zw := zip.NewWriter(out)
	defer zw.Close()

	var translated, reused, passed int

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

		upscaled, err := up.Upscale(img)
		if err != nil {
			log.Printf("%s: 放大失败 (%v)，原样复制", prefix, err)
			if err := writeEntry(f.Name, f.Mode(), f.Modified, raw); err != nil {
				return 0, 0, 0, err
			}
			passed++
			continue
		}

		outName, data, err := encodeOutput(f.Name, upscaled, cfg)
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
		translated++
	}
	return translated, reused, passed, nil
}

// ---------------------------------------------------------------------------
// Preview mode
// ---------------------------------------------------------------------------

type previewPage struct {
	Index   int
	Name    string
	OrigRel string
	TrRel   string
}

func runPreview(cfg Config, entries []entry, total int, selection []bool, cache *PageCache, up *VulkanUpscaler) (int, int, int, error) {
	origDir := filepath.Join(cfg.OutDir, "original")
	trDir := filepath.Join(cfg.OutDir, "translated")
	cmpDir := filepath.Join(cfg.OutDir, "compare")

	if !cfg.NoOrig {
		if err := os.MkdirAll(origDir, 0o755); err != nil {
			return 0, 0, 0, err
		}
	}
	if err := os.MkdirAll(trDir, 0o755); err != nil {
		return 0, 0, 0, err
	}
	if cfg.Compare {
		if err := os.MkdirAll(cmpDir, 0o755); err != nil {
			return 0, 0, 0, err
		}
	}

	var pages []previewPage
	var translated, reused, passed int

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
			return 0, 0, 0, fmt.Errorf("读取 %s: %w", f.Name, err)
		}

		if e.index == 0 {
			if !cfg.NoOrig {
				if err := writeFile(origDir, f.Name, raw); err != nil {
					return 0, 0, 0, err
				}
			}
			continue
		}

		prefix := fmt.Sprintf("[%d/%d] %s", e.index, total, f.Name)

		if !cfg.NoOrig {
			if err := writeFile(origDir, f.Name, raw); err != nil {
				return 0, 0, 0, err
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
					upscaled, err := up.Upscale(img)
					if err != nil {
						log.Printf("%s: 放大失败 (%v)", prefix, err)
						outName, outData = f.Name, raw
						passed++
					} else {
						name, data, err := encodeOutput(f.Name, upscaled, cfg)
						if err != nil {
							return 0, 0, 0, fmt.Errorf("编码 %s: %w", f.Name, err)
						}
						if err := cache.put(e.index, f.Name, name, data); err != nil {
							log.Printf("%s: 缓存写入失败: %v", prefix, err)
						}
						outName, outData = name, data
						log.Printf("%s: 完成，耗时 %.1fs", prefix, time.Since(start).Seconds())
						translated++
					}
				}
			}
		}

		if err := writeFile(trDir, outName, outData); err != nil {
			return 0, 0, 0, err
		}

		if cfg.Compare {
			composite, err := makeCompare(raw, outData)
			if err != nil {
				log.Printf("%s: 对比图生成失败: %v", prefix, err)
			} else {
				base := strings.TrimSuffix(filepath.Base(outName), filepath.Ext(outName)) + ".jpg"
				rel := filepath.ToSlash(filepath.Join(filepath.Dir(outName), base))
				if err := writeFile(cmpDir, rel, composite); err != nil {
					return 0, 0, 0, err
				}
			}
		}

		pages = append(pages, previewPage{
			Index:   e.index,
			Name:    f.Name,
			OrigRel: filepath.ToSlash(f.Name),
			TrRel:   filepath.ToSlash(outName),
		})
	}

	if !cfg.NoHTML {
		if err := writeGallery(cfg, pages); err != nil {
			return 0, 0, 0, fmt.Errorf("生成 HTML: %w", err)
		}
	}
	return translated, reused, passed, nil
}

// ---------------------------------------------------------------------------
// Compare composite
// ---------------------------------------------------------------------------

func makeCompare(origBytes, transBytes []byte) ([]byte, error) {
	orig, _, err := image.Decode(bytes.NewReader(origBytes))
	if err != nil {
		return nil, err
	}
	trans, _, err := image.Decode(bytes.NewReader(transBytes))
	if err != nil {
		return nil, err
	}

	ob, tb := orig.Bounds(), trans.Bounds()
	origH := ob.Dy()
	targetH := tb.Dy()
	if origH != targetH {
		ratio := float64(targetH) / float64(origH)
		newW := int(float64(ob.Dx()) * ratio)
		scaled := image.NewRGBA(image.Rect(0, 0, newW, targetH))
		xdraw.CatmullRom.Scale(scaled, scaled.Bounds(), orig, ob, xdraw.Over, nil)
		orig = scaled
		ob = orig.Bounds()
	}

	const gap = 20
	W := ob.Dx() + gap + tb.Dx()
	H := targetH
	dst := image.NewRGBA(image.Rect(0, 0, W, H))
	draw.Draw(dst, dst.Bounds(), image.NewUniform(color.RGBA{25, 25, 25, 255}), image.Point{}, draw.Src)
	draw.Draw(dst, image.Rect(0, 0, ob.Dx(), ob.Dy()), orig, ob.Min, draw.Src)
	draw.Draw(dst, image.Rect(ob.Dx()+gap, 0, ob.Dx()+gap+tb.Dx(), tb.Dy()), trans, tb.Min, draw.Src)

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, dst, &jpeg.Options{Quality: 85}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// ---------------------------------------------------------------------------
// Cache
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
// Zip / image helpers
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

func encodeOutput(origName string, img image.Image, cfg Config) (string, []byte, error) {
	ext := strings.ToLower(filepath.Ext(origName))
	outName := origName
	var buf bytes.Buffer
	switch ext {
	case ".jpg", ".jpeg":
		if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: cfg.JPEGQuality}); err != nil {
			return "", nil, err
		}
	case ".webp":
		if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: cfg.JPEGQuality}); err != nil {
			return "", nil, err
		}
		outName = strings.TrimSuffix(outName, filepath.Ext(outName)) + ".jpg"
	default:
		if err := png.Encode(&buf, img); err != nil {
			return "", nil, err
		}
	}
	return outName, buf.Bytes(), nil
}

// ---------------------------------------------------------------------------
// Page selection
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
// Cache listing
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
// HTML gallery
// ---------------------------------------------------------------------------

func writeGallery(cfg Config, pages []previewPage) error {
	type item struct {
		Name  string `json:"name"`
		Orig  string `json:"orig"`
		Trans string `json:"trans"`
	}
	items := make([]item, 0, len(pages))
	for _, p := range pages {
		items = append(items, item{
			Name:  p.Name,
			Orig:  "original/" + p.OrigRel,
			Trans: "translated/" + p.TrRel,
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
  #bar .badge.trans { background: #2d4a2d; color: #b8e6b8; }
  #bar .badge.orig  { background: #4a3d2d; color: #e6d0a8; }
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
  let mode = "trans";
  const $img   = document.getElementById("img");
  const $info  = document.getElementById("info");
  const $badge = document.getElementById("badge");
  const $btnT  = document.getElementById("btn-toggle");

  function render() {
    if (!pages.length) return;
    const p = pages[i];
    $img.src = (mode === "orig") ? p.orig : p.trans;
    $info.textContent = (i+1) + " / " + pages.length;
    $badge.textContent = mode === "orig" ? "原图" : "放大图";
    $badge.className = "badge " + mode;
    $btnT.textContent = mode === "orig" ? "显示放大图" : "显示原图";
    document.title = p.name + " — " + (i+1) + "/" + pages.length;
  }
  function setMode(m) { mode = m; render(); }
  function toggle()   { setMode(mode === "trans" ? "orig" : "trans"); }
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
      case "t": case "T": setMode("trans"); break;
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
// Open in file manager
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
