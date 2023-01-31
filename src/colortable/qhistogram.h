#ifndef QHistogram_h
#define QHistogram_h
#include <QVector2D>
class QHistogram
{
public:
	static const int HISTOGRAM_SIZE=256;

	QHistogram();
	QHistogram(const  QHistogram & par);
	QHistogram& operator=(const QHistogram& par);

	double& operator[](int pos);
	const double& operator[](int pos) const;
	void setRange(const QVector2D &r);
	void setValues(const double* values);

	QVector2D range() const;
private:
	double m_vals[HISTOGRAM_SIZE];
	QVector2D m_range;
};
#endif
