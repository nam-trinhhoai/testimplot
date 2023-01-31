/*
 * Matrix2D.cpp
 *
 *  Created on: 11 janv. 2018
 *      Author: j0483271
 */


template <typename T>
Matrix2DLine<T>::Matrix2DLine(int w, int h) : tab(w*h){
	this->w = w;
	this->h = h;
}

template <typename T>
Matrix2DLine<T>::~Matrix2DLine() {}

template <typename T>
const T Matrix2DLine<T>::get(unsigned int x, unsigned int y) {
	return tab[x + y*w];
}

template <typename T>
void Matrix2DLine<T>::set(const T val, unsigned int x, unsigned int y) {
	tab[x + y*w] = val;
}

template <typename T>
const double Matrix2DLine<T>::getDouble(unsigned int x, unsigned int y) {
	return (double) get(x,y);
}

template <typename T>
T* Matrix2DLine<T>::getLine(unsigned int y) {
	return &tab[y*w];
}

template <typename T>
T* Matrix2DLine<T>::getData() {
	return tab.data();
}

template <typename T>
const unsigned int Matrix2DLine<T>::width() {
	return w;
}

template <typename T>
const unsigned int Matrix2DLine<T>::height() {
	return h;
}

template <typename T>
const ImageFormats::QSampleType Matrix2DLine<T>::getType() {
    return ImageFormats::QSampleType::fromTemplate<T>();
}

template <typename T>
void* Matrix2DLine<T>::data() {
	return getData();
}

template <typename T>
void Matrix2DLine<T>::transpose() {
    std::vector<T> newTab(h*w);
    for (int j=0; j<h; j++) {
        for (int i=0; i<w; i++) {
            newTab[j+i*h] = tab[i+j*w];
        }
    }
    tab = newTab;
    int tmp = w;
    w = h;
    h = tmp;
}
