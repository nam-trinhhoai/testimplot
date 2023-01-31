#ifndef WellBoreLayer3D_H
#define WellBoreLayer3D_H

#include <QVector3D>
#include <QMatrix4x4>

#include <QColor>
#include "graphic3Dlayer.h"
#include "itooltipprovider.h"
#include "wellbore.h" // for WellUnit
#include <Qt3DRender/QPickTriangleEvent>
#include <QPhongMaterial>

class WellBoreRepOn3D;
class WellBore;

namespace Qt3DCore {
	class QEntity;
	class QTransform;

}

namespace Qt3DRender {
	class QCamera;
	class QMaterial;
}


class Vector3dD{
public:
	double m_x,m_y,m_z;

	Vector3dD()
	{
		m_x=m_y=m_z=0.0;
	}

	Vector3dD(double x,double  y, double z)
	{
		m_x=x;
		m_y=y;
		m_z=z;
	}

	Vector3dD multiply( QMatrix4x4 m)
	{
		float* data = m.data();
		double x = data[0] * m_x+ data[4]*m_y+data[8]*m_z+data[12];
		double y = data[1] * m_x+ data[5]*m_y+data[9]*m_z+data[13];
		double z = data[2] * m_x+ data[6]*m_y+data[10]*m_z+data[14];
		double w = data[3] * m_x+ data[7]*m_y+data[11]*m_z+data[15];
		x =x/w;
		y =y/w;
		z =z/w;
		return Vector3dD(x,y,z);
	}

	QVector3D convert()
	{
		return QVector3D(m_x,m_y,m_z);
	}

};

class WellBoreLayer3D : public Graphic3DLayer, public IToolTipProvider {
Q_OBJECT
public:
	typedef struct Container {
		Qt3DCore::QEntity* entity;
		Qt3DCore::QEntity* entitylog;
		Qt3DCore::QTransform* transform;
		Qt3DExtras::QPhongMaterial* mat;
	} Container;

	WellBoreLayer3D(WellBoreRepOn3D *rep, QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera);
	virtual ~WellBoreLayer3D();

	virtual void show() override;
	virtual void hide() override;

	virtual QRect3D boundingRect() const override;

	virtual void refresh() override;
	virtual void zScale(float val) override;

	bool isShown() const;

	void updateLog();
	void showLog();

	double distanceSimplification()
	{
		return m_distanceSimplification;
	}

	void setDefaultWidth(long defaultWidth);
	void setMinimalWidth(long minimalWidth);
	void setMaximalWidth(long maximalWidth);
	void setLogMin(double logMin);
	void setLogMax(double logMax);
	void setDefaultColor(QColor defaultColor);

	void selectWell(QString ,int,int, QVector3D );
	void deselectWell();

	void deselectLastWell();
	void hideLastWell();

	virtual QString generateToolTipInfo() const override;

signals:
	void layerShownChanged(bool toggle);
	void showNameSignal(const IToolTipProvider*,QString,int,int,QVector3D);
	void hideNameSignal();
	void selectSignal(WellBoreLayer3D*);


public slots:
	void setDistanceSimplification(double value);
	void setIncrementLogs(int value);

	void setWireframe(bool);
	void setShowNormals(bool);

	void setThicknessLog(int);
	void setColorLog(QColor );

	void setColorWell(QColor);
	void setColorSelectedWell(QColor);

	void setDiameterWell(int);

private:
	WellBore * wellBore() const;

	QVector3D applyPalette(float ratio) const;
//	QVector3D getPosFromMd(double md, bool* ok) const;
	Vector3dD getPosFromWellUnitD(double unitIndex, WellUnit wellUnit, bool* ok) const;



	WellBoreRepOn3D* m_rep;
    Qt3DRender::QCamera *m_camera;
	QList<Container> m_lineEntities;
	bool m_isShown;
	QVector3D m_colorA, m_colorB;

	long m_defaultWidth = 30;
	long m_minimalWidth = 50;
	long m_maximalWidth = 90;
	double m_logMin = -1;
	double m_logMax = 1;
	QColor m_defaultColor = QColor(255, 255, 0); //QColor(128, 0, 255);
	QColor m_selectedColor = QColor(0, 255, 255);

	QColor m_colorLog = Qt::green;
	int m_thicknessLog=3;
	bool m_selected;

	int m_incrLog =1;
	double m_distanceSimplification = 2.0;
	bool m_modeWireframe = false;
	bool m_showNormals= false;
};

#endif
