#include "affinetransformation.h"

AffineTransformation::AffineTransformation(double a, double b, QObject *parent) :
		QObject(parent) {
	m_direct = std::array<double, 2> { a, b };
}

AffineTransformation::AffineTransformation(QObject *parent) :
		QObject(parent) {
	m_direct = std::array<double, 2> { 1, 0 };
}

AffineTransformation::AffineTransformation(const AffineTransformation &par) {
	this->m_direct = par.m_direct;
}

AffineTransformation& AffineTransformation::operator=(
		const AffineTransformation &par) {
	if (this != &par) {
		this->m_direct = par.m_direct;
	}
	return *this;
}

AffineTransformation::~AffineTransformation() {

}

void AffineTransformation::direct(double world, double &image) const {
	image = m_direct[0] * world + m_direct[1];
}
void AffineTransformation::indirect(double image, double &world) const {
	world = (image - m_direct[1]) / m_direct[0];
}
