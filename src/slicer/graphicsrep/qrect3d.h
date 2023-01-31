#ifndef QRect3D_H
#define QRect3D_H

class QRect3D
{
public:
	QRect3D(double x, double y, double z,double width, double height, double depth);
	QRect3D();

	QRect3D(const  QRect3D & );
	QRect3D& operator=(
				const QRect3D&);

	 inline double x() const{return m_x;}
	 inline double y() const{return m_y;}
	 inline double z() const{return m_z;}

	 inline double width() const{return m_width;}
	 inline double height() const{return m_height;}
	 inline double depth() const{return m_depth;}

	bool merge(const QRect3D & in);

	virtual ~QRect3D();

protected:

protected:
	double m_x,m_y, m_z;
	double m_width, m_height,m_depth;
	bool m_valid = true;
};

#endif
