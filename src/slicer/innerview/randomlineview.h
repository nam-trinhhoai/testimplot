#ifndef RANDOMLINEVIEW_H_
#define RANDOMLINEVIEW_H_

#include "abstract2Dinnerview.h"
#include "affinetransformation.h"
#include <utility>
#include <QVector2D>
#include <QObject>
#include <QTimer>
#include <QGraphicsLineItem>
#include "GraphEditor_CurveShape.h"
#include "GraphEditor_RegularBezierPath.h"
#include "GraphEditor_ListBezierPath.h"
#include "GraphEditor_CrossItem.h"


class Affine2DTransformation;
class QGLGridTickItem;
class QGLCrossItem;
class QGLFixedAxisItem;
class IGeorefImage;
class RandomRep;
class SeismicSurvey;
class GraphEditor_LineShape;
class GraphEditor_ItemInfo;
class RandomLineViewOverlayPainter; // Local
class QSlider;
class QSpinBox;
class QLineEdit;
class RandomTransformation;


//Specialized graphic view to handle section: all the views need to be synchronized
class RandomLineView : public Abstract2DInnerView {
	Q_OBJECT
	friend class RandomLineViewOverlayPainter;
public:

	QVector3D m_position3DCross;
	QPointF m_position2DCross;
	IsoSurfaceBuffer  m_isoBuffer;

	RandomLineView(QPolygonF randomPolyLine, ViewType type, QString uniqueName,eRandomType eType=eTypeStandard);
	 ~RandomLineView() ;

	QPolygonF polyLine() const {
		return m_polyLine;
	}

	// point is defined on grid
	QPolygon discreatePolyLine() const {
		return m_discreatePolyLine;
	}

	QPolygonF worldDiscreatePolyLine() const {
		return m_worldDiscreatePolyLine;
	}

	void showRep(AbstractGraphicRep * rep) override;
	void hideRep(AbstractGraphicRep *rep) override;

	RandomTransformation* randomTransform();

	RandomRep* firstRandom() const;
	RandomRep* lastRandom() const;

	RandomRep* getRandomSismic() const;

	QVector<RandomRep*> getRandomVisible();

	QVector3D viewWorldTo3dWordExtended(QPointF posi);

	QPointF absoluteWorldToViewWorldExtended(QVector3D pos);
	// return the index if point inside the polyline or -1 if it is outside
	long getDiscreatePolyLineIndexFromScenePos(QPointF);
	QPointF getScenePosFromDiscreatePolyLinePos(QPointF);
	std::tuple<long, double, bool> getDiscreatePolyLineIndexFromWorldPos(QPointF worldPos);

	double displayDistance() const;
	void setDisplayDistance(double);
	double getWellSectionWidth() const;
	void setWellSectionWidth(double value);
	void defineSliceMinMax(int departX, int width) ;//IGeorefImage *image);
	void updateSlicePosition(int worldVal);
	void setOrthogonalView(RandomLineView *pOrthoView);
	RandomLineView * getOrthogonalView(void);
	QPolygon  getOrthoDiscreatePolyline() { return m_discreateOrthogonalPolyLine; }
    QPolygonF getOrthoWorldDiscreatePolyline() { return m_worldDiscreateOrthogonalPolyLine; }
    void setOrthoWorldDiscreatePolyline(QPolygonF line){ m_worldDiscreateOrthogonalPolyLine = line;}
    void setOrthoDiscreatePolyline(QPolygon line){ m_discreateOrthogonalPolyLine = line;}
    void addOrthogonalLine(GraphEditor_LineShape* pLine);
    void setItemSection(GraphEditor_ItemInfo* pItem);
    eRandomType getRandomType();

    void discreatePoly(  Seismic3DAbstractDataset *dataset);

    void initTransformation();

    static void discreatePoly(QVector<QPointF> polyLine,  Seismic3DAbstractDataset *dataset,QPolygon& discreatePolyLine , QPolygonF& worldDiscreatePolyLine);
    int sizeDiscretePolyline(){
    		return m_discreatePolyLine.count();
    	}

    GraphEditor_CurveShape* getCurveItem(){
    	return m_curveItem;
    }

    QVector3D viewWorldTo3dWord(QPointF);
    QPointF absoluteWorldToViewWorld(QVector3D pos);

    void setColorCross(QColor);
    void setCrossPosition(QPointF pos);

    void setSpeedAnimation(int);

    void refreshOrtho(QVector3D,QVector3D);


    void ResetLineOrtho()
    {
    	m_orthoLineList.clear();

    }

    void setCurrentNaneNurbs(QString s)
    {
    	m_currentName = s;
    }
    void showEvent(QShowEvent* event) override;

    bool isAnimActif() { return m_animActif;}



    GraphEditor_Path* getCurveBezier(){ return m_curveBezier;}
    GraphEditor_Path* getBezierItem(){ return m_bezierItem;}

    void setBezierItem(GraphEditor_ListBezierPath * path)
    {
    	m_bezierItem =path;
    }

    void setCurveBezier(int slice, GraphEditor_Path* bezier){m_numSlice = slice; m_curveBezier = bezier;}


    void deleteCurves();
  /*  QPointF getPointOrtho()
    {
    	return m_pointOthogonal;
    }*/

    GraphEditor_Path* m_BezierDirectrice = nullptr;


    int getSliceCurrent() { return m_sliceValueWorld;}

protected slots:

	void onTimeout();
	void playMoveSection();
	void playInverseMoveSection();
	void nextSection();
	void previousSection();
	void sliceFinish();
	void sliceChanged(int val);
	void sliceFinishChanged(int val);
	void orthogonalMoved(int val);
	void setPolyLine(QPolygonF polyLine);
	void updateOrthogonalWidth(const QString &newText);

	void nurbsXYZChanged(QVector3D normal);
	void nurbYChanged(float);
	void curveChanged(std::vector<QVector3D>, bool isopen =true);
	void curveChangedTangent(QVector<PointCtrl> listepts3d, bool isopen,QPointF cross);
	void curveChangedTangent2(QVector<QVector3D> listepts3d,QVector<QVector3D> listetan3d, bool isopen,QPointF cross,QString);
	void curveChangedTangentOpt(GraphEditor_ListBezierPath* path);
	void receiveCurrentIndex(int);


	void addFromCross(QPointF);
	void moveCross(int,int,int);
	void positionCross(int,QPointF);

	void resetItemSection();

signals:
	void nextKeySection(float);

	void etatAnimationChanged(bool);
	void updateOrthoFrom3D(QVector3D,QPointF);
	void updateOrtho(QPolygonF);
	void newWidthOrtho(QPolygonF);

	void signalAddPointsDirectrice(QPointF);
	void signalMovePointsDirectrice(int,int,int);
	void signalPositionPointsDirectrice(int,QPointF);
	void signalCurrentIndex(int);
	void signalMoveCrossFinish();

	void moveCamFollowSection(int);
	void sendIndexChanged(int);
	void createXSection3D();

	void displayDistanceChanged(double);
	void orthogonalSliceMoved(int,int,QString);
	void newPointPosition(QPointF);
	void newWidthOrthogonal(double);
	void linedeteled(GraphEditor_LineShape*);

	void signalWellSectionWidth(double);

	void select(QString);

	void destroyed();



protected:
	bool absoluteWorldToViewWorld(MouseTrackingEvent &event) override;
	bool viewWorldToAbsoluteWorld(MouseTrackingEvent &event) override;

	void addAxis(IGeorefImage *image);
	//void defineScale(RandomRep *rep);
	void removeAxis();

	void contextualMenuFromGraphics(double worldX, double worldY, QContextMenuEvent::Reason reason, QMenu& mainMenu) override;

	void cleanupRep(AbstractGraphicRep *rep) override;

	void updateTitleFromSlices();

	virtual void setDepthLengthUnitProtected(const MtLengthUnit* depthLengthUnit) override;
	void updateAxisFromLengthUnit();

public slots:
	void showValues(bool checked);
	void receiveCurrentGrabberIndex(int);
	void moveCrossFinish();

	void moveLineOrtho(QPointF ,QPointF);




protected slots:
	void mainSceneRectChanged();


protected:
	QGLFixedAxisItem * m_verticalAxis;
	QGLFixedAxisItem * m_horizontalAxis;

	GraphEditor_CrossItem* m_crossItem = nullptr;



	//QGraphicsLineItem * m_lineVItem = nullptr;
	//QGraphicsLineItem * m_lineHItem = nullptr;
	int m_sizeCross= 15;
	QRectF m_rectIntersect;




//	int m_currentSliceIJPosition;
//	int m_currentSliceWorldPosition;

private:
	int getAltitudeCross();

	float getDistance(int index,QVector<QPointF> keyPoints);
	QWidget* createSliceBox(const QString &title);
	void defineSliceVal(int image);
	void getPositionMap(int position);
	void  computeDiscreateData(Seismic3DAbstractDataset *dataset);
	RandomLineViewOverlayPainter* m_overlayPainter = nullptr;
////	LayerSpectrumDialog* m_layerSpectrumDialog = nullptr;
//
	double m_contextualWorldX = 0;
	double m_contextualWorldY = 0;
	bool m_isShowValues = false;
//
	QPolygonF m_polyLine;
	QPolygon m_discreatePolyLine;
	QPolygonF m_worldDiscreatePolyLine;

	QPolygon m_discreateOrthogonalPolyLine;
	QPolygonF m_worldDiscreateOrthogonalPolyLine;

	SeismicSurvey* m_supportSurvey = nullptr;
	long m_nbDatasetWidth, m_nbDatasetDepth;
	SampleUnit m_randomType;
//	std::map<std::size_t, std::size_t> m_nodePositionInDiscreate; // may be needed one day
//	std::vector<QString> *m_horizonNames = nullptr;
//	std::vector<QString> *m_horizonPaths = nullptr;
	double m_displayDistance = 100;
	QToolButton* m_playButton;
	QToolButton* m_playButtonInverse;
	QSlider *m_sliceImageSlider;
	QSpinBox *m_sliceSpin;
	QLineEdit *m_Orthogonalwidth;
	int m_sliceValueWorld = 0;
	int m_OrthogonalWidth;
	RandomLineView *m_RandomOrthogonal;
	eRandomType m_eType;
	RandomRep *m_slice;
	QPointF m_pointOthogonal;
	bool m_UpdateRep;
	QList<GraphEditor_LineShape *> m_orthoLineList;
	GraphEditor_ItemInfo* m_ItemSection;

	GraphEditor_CurveShape *m_curveItem= nullptr;
	QPointer<GraphEditor_ListBezierPath> m_bezierItem = nullptr;


	GraphEditor_Path* m_curveBezier = nullptr;

	RandomTransformation *m_transformation;



	QTimer* mTimer;
	float m_coef = 1.0f;
	int m_stepsAnimation = 1;
	bool m_animActif = false;
	int m_sensAnim= 1;


	int m_numSlice =0;
	int m_WidthOriginal;
	QPointF m_posTemp;
	float lastValue =3000.0f;

	QString m_currentName="";

	// naming function cache
	RandomRep* m_cacheSliceNameObj = nullptr;
	bool m_cacheIsNameRgt = false;

	// common value for wells
	double m_wellSectionWidth = 2.0;


};

#endif
