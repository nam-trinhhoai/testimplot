#ifndef QGLAbstractFullImage_H_
#define QGLAbstractFullImage_H_

#include "qglabstractimage.h"

class QGLAbstractFullImage: public QGLAbstractImage {
Q_OBJECT
public:
	QGLAbstractFullImage(QObject *parent = 0);
	~QGLAbstractFullImage();

	virtual void bindTexture(unsigned int unit)=0;
	virtual void releaseTexture(unsigned int unit)=0;
};

#endif /* QTLARGEIMAGEVIEWER_QGLSIMPLEIMAGE_H_ */
