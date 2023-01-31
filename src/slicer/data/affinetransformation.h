#ifndef AffineTransformation_H
#define AffineTransformation_H

#include <QObject>
#include "igeorefimage.h"
#include <array>

class AffineTransformation: public QObject
{
Q_OBJECT
public:
	AffineTransformation(double a,double b, QObject * parent=nullptr);
	AffineTransformation(QObject * parent=nullptr);

		//Copy constructor
	AffineTransformation(const  AffineTransformation & );
	AffineTransformation& operator=(
				const AffineTransformation&);

	~AffineTransformation();

	virtual void direct(double world,double &image) const ;
	virtual void indirect(double image, double &world)const  ;

	double a() const{return m_direct[0];}
	double b() const{return m_direct[1];}
private:
	std::array<double,2> m_direct;

};

#endif
