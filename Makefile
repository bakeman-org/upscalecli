# 结果：assets/realesrgan-ncnn-vulkan  和  assets/models/*
prepare-res:
	rm realesrgan-ncnn-*.zip*
	mkdir -p assets
	wget https://gh-proxy.org/https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip
	unzip -d assets realesrgan-ncnn-vulkan-20220424-ubuntu.zip