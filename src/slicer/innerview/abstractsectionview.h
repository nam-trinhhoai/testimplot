#ifndef AbstractSectionView_H
#define AbstractSectionView_H

#include "abstract2Dinnerview.h"
#include "affinetransformation.h"
#include "viewutils.h"
#include <QVector2D>

class Affine2DTransformation;
class SliceRep;
class QSlider;
class QSpinBox;
class QGroupBox;
class QPushButton;
class QGLGridTickItem;
class QGLFixedAxisItem;
class IGeorefImage;

class SyncViewer2dOverlayPainter; // Local
class LayerSpectrumDialog;

//Specialized graphic view to handle section: all the views need to be synchronized
class AbstractSectionView: public Abstract2DInnerView {
Q_OBJECT
friend class SyncViewer2dOverlayPainter;
public:
	AbstractSectionView(bool restictToMonoTypeSplit,ViewType type, 
	QString uniqueName);
	virtual ~AbstractSectionView();

	virtual void updateSlicePosition(int worldVal, int imageVal);
	int getCurrentSliceWorldPosition() const;

	bool isMapRelationSet() const;
	QPointF getPointOnMap(int worldVal) const;
	std::pair<QPointF, QPointF> getSectionSegment() const;

	const Affine2DTransformation* inlineXLineToXY() const;

	SliceRep * firstSlice() const;
	SliceRep * lastSlice() const;

	double displayDistance() const;
	void setDisplayDistance(double);

	double getWellSectionWidth() const;
	void setWellSectionWidth(double value);

	QVector3D viewWorldTo3dWord(QPointF);

signals:
	void displayDistanceChanged(double);
	void signalWellSectionWidth(double);

protected:
	virtual bool absoluteWorldToViewWorld(MouseTrackingEvent &event) override;
	virtual bool viewWorldToAbsoluteWorld(MouseTrackingEvent &event) override;

	void addAxis(IGeorefImage *image);
	void defineScale(SliceRep *rep);
	void removeAxis();

	virtual void contextualMenuFromGraphics(double worldX, double worldY, QContextMenuEvent::Reason reason, QMenu& mainMenu) override;
	void toggleXAxis(bool toggle);
	QIcon getOrientationIcon();

	void updateTitleFromSlices();

	virtual void setDepthLengthUnitProtected(const MtLengthUnit* depthLengthUnit) override;
	void updateAxisFromLengthUnit();
	void updateVerticalAxisColor();

protected slots:
//	void spe<ctrumDecomposition();
	void showValues(bool checked);
	void toggleOrientationUI();
	void mainSceneRectChanged();

protected:
	QGLFixedAxisItem * m_verticalAxis;
	QGLFixedAxisItem * m_horizontalAxis;

	int m_currentSliceIJPosition;
	int m_currentSliceWorldPosition;

	SampleUnit m_sectionType = SampleUnit::NONE;
private:

	SyncViewer2dOverlayPainter* m_overlayPainter = nullptr;
//	LayerSpectrumDialog* m_layerSpectrumDialog = nullptr;

	double m_contextualWorldX = 0;
	double m_contextualWorldY = 0;
	bool m_isShowValues = false;
//	std::vector<QString> *m_horizonNames = nullptr;
//	std::vector<QString> *m_horizonPaths = nullptr;
	double m_displayDistance = 100;

	bool m_isXAxisReversed = false;
	QPushButton* m_orientationButton;

	// naming function cache
	SliceRep* m_cacheSliceNameObj = nullptr;
	bool m_cacheIsNameRgt = false;

	// common value for wells
	double m_wellSectionWidth = 2.0;
};

#endif
