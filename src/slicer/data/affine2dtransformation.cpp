#include "affine2dtransformation.h"

#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <gdal.h>
using namespace boost::numeric::ublas;

Affine2DTransformation::Affine2DTransformation(int width, int height, const std::array<double,6> &direct,QObject * parent):QObject(parent)
{
	m_width = width;
	m_height = height;
	m_direct = direct;

	double geotransform[6];
	for(int i=0;i<6;i++)
		geotransform[i]= direct[i];

	double reverse[6];
	GDALInvGeoTransform(geotransform,reverse);

	for(int i=0;i<6;i++)
		m_inverse[i]= reverse[i];


	//	matrix<double> C(3, 3);
	//	C(0, 0) = m_direct[1];
	//	C(0, 1) = m_direct[2];
	//	C(0, 2) = m_direct[0];
	//
	//	C(1, 0) = m_direct[4];
	//	C(1, 1) = m_direct[5];
	//	C(1, 2) = m_direct[3];
	//
	//	C(2, 0) = 0;
	//	C(2, 1) = 0;
	//	C(2, 2) = 1;
	//
	//	//brute force approach to get the inverse
	//	matrix<double> CInv = identity_matrix<float>(C.size1());
	//	permutation_matrix<size_t> pm(C.size1());
	//	lu_factorize(C, pm);
	//	lu_substitute(C, pm, CInv);
	//
	//	//fill m_inverse!
	//	m_inverse[1] = CInv(0, 0);
	//	m_inverse[2] = CInv(0, 1);
	//	m_inverse[0] = CInv(0, 2);
	//
	//	m_inverse[4] = CInv(1, 0);
	//	m_inverse[5] = CInv(1, 1);
	//	m_inverse[3] = CInv(1, 2);
}


Affine2DTransformation::Affine2DTransformation(int width, int height,QObject * parent):QObject(parent) {
	m_width = width;
	m_height = height;

	m_direct[0] = m_inverse[0] = 0;
	m_direct[1] = m_inverse[1] = 1;
	m_direct[2] = m_inverse[2] = 0;

	m_direct[3] = m_inverse[3] = 0;
	m_direct[4] = m_inverse[4] = 0;
	m_direct[5] = m_inverse[5] = 1;
}


Affine2DTransformation::Affine2DTransformation(const  Affine2DTransformation & par)
{
	this->m_width = par.m_width;
	this->m_height = par.m_height;
	this->m_direct = par.m_direct;
	this->m_inverse = par.m_inverse;
}

Affine2DTransformation& Affine2DTransformation::operator=(
			const Affine2DTransformation& par)
{
	if (this != &par) {
		this->m_width = par.m_width;
		this->m_height = par.m_height;
		this->m_direct = par.m_direct;
		this->m_inverse = par.m_inverse;
	}
	return *this;
}

Affine2DTransformation::~Affine2DTransformation() {

}

std::array<double,6> Affine2DTransformation::direct() const
{
	return m_direct;
}
std::array<double,6> Affine2DTransformation::inverse() const
{
	return m_inverse;
}


void Affine2DTransformation::applyTransform(const std::array<double, 6> &transfo,
		double i, double j, double &x, double &y) const {
	x = transfo[1] * i + transfo[2] * j + transfo[0];
	y = transfo[4] * i + transfo[5] * j + transfo[3];
}

void Affine2DTransformation::worldToImage(double worldX, double worldY,
		double &imageX, double &imageY) const {
	applyTransform(m_inverse, worldX, worldY, imageX, imageY);

}
void Affine2DTransformation::imageToWorld(double imageX, double imageY,
		double &worldX, double &worldY) const {
	applyTransform(m_direct, imageX, imageY, worldX, worldY);
}
QMatrix4x4 Affine2DTransformation::imageToWorldTransformation() const {
	return QMatrix4x4((float) m_direct[1], (float) m_direct[2], 0.0f,
			(float) m_direct[0], (float) m_direct[4], (float) m_direct[5], 0.0f,
			(float) m_direct[3], 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
}

int Affine2DTransformation::width() const {
	return m_width;
}
int Affine2DTransformation::height() const {
	return m_height;
}

QRectF Affine2DTransformation::worldExtent() const {
	return IGeorefImage::worldExtent(this);
}
bool Affine2DTransformation::valueAt(int i, int j, double &value) const {
	//Ignore me
	return false;
}

void Affine2DTransformation::valuesAlongJ(int j, bool *valid,
		double *values) const {
	//Ignore me
}
void Affine2DTransformation::valuesAlongI(int i, bool *valid,
		double *values) const {
	//Ignore me
}
