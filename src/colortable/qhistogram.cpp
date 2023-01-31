#include "qhistogram.h"
#include <iostream>
#include <cstdint>
#include <cstring>

const int QHistogram::HISTOGRAM_SIZE;

QHistogram::QHistogram() {
	this->m_range = QVector2D(0, 0);
	std::memset(this->m_vals, 0, sizeof(double) * QHistogram::HISTOGRAM_SIZE);
}
QHistogram::QHistogram(const QHistogram &par) {
	this->m_range = par.m_range;
	std::memcpy(this->m_vals, par.m_vals,
			sizeof(double) * QHistogram::HISTOGRAM_SIZE);
}

QHistogram& QHistogram::operator=(const QHistogram &par) {
	if (this != &par) {
		this->m_range = par.m_range;
		std::memcpy(this->m_vals, par.m_vals,
				sizeof(double) * QHistogram::HISTOGRAM_SIZE);
	}
	return *this;
}

double& QHistogram::operator[](int pos) {
	return m_vals[pos];
}

const double& QHistogram::operator[](int pos) const {
	return m_vals[pos];
}

void QHistogram::setValues(const double *values) {
	std::memcpy(this->m_vals, values,
			sizeof(double) * QHistogram::HISTOGRAM_SIZE);
}

void QHistogram::setRange(const QVector2D &r) {
	m_range = r;
}

QVector2D QHistogram::range() const {
	return m_range;
}
