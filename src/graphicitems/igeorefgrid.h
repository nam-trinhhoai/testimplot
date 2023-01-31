#ifndef IGeorefGrid_H
#define IGeorefGrid_H

#include <QMatrix4x4>

class IGeorefGrid {
public:
	IGeorefGrid();
	virtual ~IGeorefGrid();

	virtual void worldToImage(double worldX, double worldY,double &imageX, double &imageY)const=0;
	virtual void imageToWorld(double imageX, double imageY,double &worldX, double &worldY)const=0;

	virtual QMatrix4x4 imageToWorldTransformation()const=0;

	virtual int width() const =0;
	virtual int height() const =0;

	virtual QRectF worldExtent() const=0;

	static QRectF worldExtent(const IGeorefGrid* const image);

	static QRectF imageToWorld(const IGeorefGrid * const image, const QRectF & rect);
private:
	static QRectF computeRect(const IGeorefGrid *const image,double * ij);
};

#endif /* IGeorefGrid_H */
