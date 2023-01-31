#ifndef Affine2DTransformation_H
#define Affine2DTransformation_H

#include <QObject>
#include "igeorefimage.h"
#include <array>

class Affine2DTransformation: public QObject, public IGeorefImage
{
Q_OBJECT
public:
	Affine2DTransformation(int width, int height, const std::array<double,6> &direct,QObject * parent=nullptr);
	Affine2DTransformation(int width, int height,QObject * parent=nullptr);

		//Copy constructor
	Affine2DTransformation(const  Affine2DTransformation & );
	Affine2DTransformation& operator=(
				const Affine2DTransformation&);

	~Affine2DTransformation();

	std::array<double,6> direct() const;
	std::array<double,6> inverse() const;


	virtual void worldToImage(double worldX, double worldY,double &imageX, double &imageY) const override;
	virtual void imageToWorld(double imageX, double imageY,double &worldX, double &worldY)const  override;

	virtual QMatrix4x4 imageToWorldTransformation()const  override;

	virtual int width() const override;
	virtual int height() const override;

	virtual QRectF worldExtent() const override;
	virtual bool valueAt(int i, int j, double &value)const override;

	virtual void valuesAlongJ(int j, bool* valid,double* values)const override;
	virtual void valuesAlongI(int i, bool* valid,double* values)const override;
private:
	void applyTransform(const std::array<double, 6>& transfo, double i,
			double j, double &x, double &y)const ;
private:
	int m_width;
	int m_height;

	std::array<double,6> m_direct;
	std::array<double,6> m_inverse;

};

#endif
