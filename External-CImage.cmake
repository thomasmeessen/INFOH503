message("External project - CImage")

ExternalProject_Add(CImg 
GIT_REPOSITORY https://github.com/dtschump/CImg.git
GIT_TAG master
SOURCE_DIR Cimg
Binaries_DIR CImg-build

)