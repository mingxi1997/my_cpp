
sudo apt install libcanberra-gtk-module libcanberra-gtk3-module

cd /home/user_name/OpenCv
mkdir Release
cd Release

cmake -DOPENCV_EXTRA_MODULES_PATH=/home/xu/opencv/opencv_contrib/modules /home/xu/opencv/opencv -D  CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_GTK=ON -D WITH_OPENGL=ON ..

make
sudo make install 
