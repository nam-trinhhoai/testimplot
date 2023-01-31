#ifndef SeismicSurveyRepOnMap_H
#define SeismicSurveyRepOnMap_H

#include <QObject>
#include <QMap>
#include <QVector2D>
#include <QPointF>
#include <QGraphicsSvgItem>
#include <QGraphicsLineItem>

#include "seismicsurveyrep.h"
#include "idatacontrolerholder.h"
#include "helicoitem.h"

class SeismicSurveyPropPanel;
class SeismicSurveyLayer;
class Seismic3DAbstractDataset;
class CameraParametersController;

class SeismicSurveyRepOnMap: public SeismicSurveyRep,
		public IDataControlerHolder {
Q_OBJECT
public:
	SeismicSurveyRepOnMap(SeismicSurvey *survey, AbstractInnerView *parent = 0);
	virtual ~SeismicSurveyRepOnMap();

	virtual bool canBeDisplayed() const override {
		return true;
	}
	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;

	//IImagePositionControlerHolder
	QGraphicsItem* getOverlayItem(DataControler *controler,
			QGraphicsItem *parent) override;
	QGraphicsItem* releaseOverlayItem(DataControler *controler) override;
	virtual void notifyDataControlerMouseMoved(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys) override;
	virtual void notifyDataControlerMousePressed(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys)override;
	virtual void notifyDataControlerMouseRelease(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys)override;
	virtual void notifyDataControlerMouseDoubleClick(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys)override;

//	void  transformItemZoomScale(QGraphicsSvgItem* item, CameraParametersController* controler, QVector3D position);
//	void  positionChanged(CameraParametersController *controler,QGraphicsSvgItem* item );

//	void showHelico(QGraphicsSvgItem* item, CameraParametersController* controler, bool b);
	virtual bool eventFilter(QObject* watched, QEvent* ev) override;
	virtual TypeRep getTypeGraphicRep() override;

private:
	Seismic3DAbstractDataset * containsDatasetID(const QUuid &uuid);

	QVector3D moyenneTab();
private:
	SeismicSurveyPropPanel *m_propPanel;
	SeismicSurveyLayer *m_layer;

	//QGraphicsLineItem * m_lineItem;
	//QGraphicsSvgItem* m_item;
	CameraParametersController* m_ctrl;

	QMap<DataControler*, QGraphicsItem*> m_datacontrolers;
	//QPointF m_lastPosition;

//	float m_posX, m_posY;

	//QVector<QVector3D> m_tabDirection;
	//int m_indexTab;
	//int m_sizeTab = 25;
	//bool m_debug;
	//bool m_helicoVisible = false;

};

#endif
