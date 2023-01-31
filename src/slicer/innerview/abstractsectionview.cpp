#include "abstractsectionview.h"

#include <sstream>
#include <iostream>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QSlider>
#include <QSpinBox>
#include <QPushButton>
#include <QMenu>
#include <QPoint>
#include <QPointF>
#include <QRect>
#include <cmath>

#include "slicerep.h"
#include "sliceqglgraphicsview.h"
#include "slicepositioncontroler.h"
#include "qglgridtickitem.h"
#include "cudaimagepaletteholder.h"
#include "mousetrackingevent.h"
#include "seismicsurvey.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "qglfixedaxisitem.h"
#include "affine2dtransformation.h"
#include "statusbar.h"
#include "qglscalebaritem.h"
#include "isliceablerep.h"
#include "LayerSpectrumDialog.h"
#include "affine2dtransformation.h"
#include "mtlengthunit.h"

struct SyncViewer2dOverlayPainter : public AbstractOverlayPainter {
private:
	AbstractSectionView* m_view2d;
public:
	SyncViewer2dOverlayPainter(AbstractSectionView* view2d) : m_view2d(view2d){
	}
	virtual ~SyncViewer2dOverlayPainter(){
	}

	QRect computeBoundingBox(const QFontMetrics& metrics, const QStringList& list, int width){
		int rwidth = 0;
		int rheight = 0;
		for(const QString& info : list){
			QString elided = metrics.elidedText(info,Qt::TextElideMode::ElideRight,width);
			QRect bounding = metrics.boundingRect(elided);
			rwidth = std::max(rwidth,bounding.width());
			rheight += bounding.height();
		}
		return QRect(0,0,rwidth,rheight);
	}

	void paintOverlay(QPainter* painter, const QRectF& rect) override{

		if ( !m_view2d->m_isShowValues )
			return;

		int halfWidth = rect.width()/2;
		painter->setPen(Qt::yellow);
		painter->setBrush(Qt::black);

		int topLeft = 5;
		int topRight = 5;
		int bottomRight = rect.height()-5;
		int valueFillX = 0;
		int valueFillY = 0;
		int valueHeightMax = 0;
		const int valueBorder = 1;

		QFontMetrics metrics = painter->fontMetrics();

		// ================== VALUES ====================================================
		//		QGraphicsView* graphicsView =  this->view2d->getScene()->views().first();
		//		QWidget* viewport = this->view2d->view->viewport();
		//		QPointF mousePosition1 = graphicsView->mapFromScene(view2d->lastMousePositionWorld.toPoint());
		//		QString valueStr = QString::number(view2d->lastMousePositionImage.x()) + "/" + QString::number(view2d->lastMousePositionImage.y());
		//
		QPointF lastMousePosition = m_view2d->getLastMousePosition();
		QPoint mapPos = m_view2d->m_view->mapFromScene(lastMousePosition);
		QPoint globalPos = m_view2d->m_view->mapToGlobal(mapPos);
		if ( lastMousePosition.x() != -1 ) {
			std::vector<QString> vect;
			for (AbstractGraphicRep *rep : m_view2d->m_visibleReps) {
				SliceRep *slice = nullptr;
				if ( ! (slice = dynamic_cast<SliceRep*>(rep))) {
					continue;
				}
				IData* ds = slice->data();
				QString name = ds->name();
				Seismic3DAbstractDataset* dataset = nullptr;
				if ( ! (dataset = dynamic_cast<Seismic3DAbstractDataset*>(ds))) {
					continue;
				}
				qDebug() << "Data Name: " << name <<
						"Data Type: " << dataset->type();

				if (IMouseImageDataProvider *provider =
						dynamic_cast<IMouseImageDataProvider*>(rep)) {
					IMouseImageDataProvider::MouseInfo info;
					if (provider->mouseData(lastMousePosition.x(), lastMousePosition.y(), info)) {
						std::stringstream ss;
						//ss << std::fixed << std::setprecision(2);
						if (info.values.size() < 1) {
							continue;
						}

						QString valueStr1 = metrics.elidedText(QString::number(info.values[0]),
								Qt::TextElideMode::ElideRight,halfWidth);
						vect.push_back(valueStr1);
						QRect boundingVal = metrics.boundingRect(valueStr1);
						if ( valueFillX < boundingVal.width() + 2 * valueBorder) valueFillX = boundingVal.width() + 2 * valueBorder;
						if ( valueHeightMax < boundingVal.width()) valueHeightMax = boundingVal.width();
						valueFillY += boundingVal.height() +2 * valueBorder;
					}
				}
			}
			// Line Pointer not actif, draw values on cursor move
			// ================== VALUES ====================================================
			int valuePosX= mapPos.x() + 5;
			int valuePosY = mapPos.y() + 5;
			painter->fillRect(valuePosX, valuePosY ,valueFillX, valueFillY, QColor(0,0,0,255));
			reverse(vect.begin(), vect.end());
			for(QString s:vect) {
				QRect boundingVal = metrics.boundingRect(s);
				painter->drawRect( valuePosX, valuePosY, valueFillX, boundingVal.height()+2*valueBorder);
				painter->drawText( valuePosX+1, valuePosY+boundingVal.height()+valueBorder,s);
				valuePosY += boundingVal.height() + 2 * valueBorder;
			}
		}
//
//		// INFO ===============================================================================================
//		for(auto pair : view2d->handleTovisuals){
//			// ======= top left
//			QStringList topLeftInfo = pair.second->topLeftInfo();
//			QRect topLeftBounding = computeBoundingBox(metrics, topLeftInfo, halfWidth);
//			painter->fillRect(0,topLeft,topLeftBounding.width()+10, topLeftBounding.height()+5,QColor(127,127,127,127));
//			for(const QString& info : topLeftInfo){
//				QString elided = metrics.elidedText(info,Qt::TextElideMode::ElideRight,halfWidth);
//				QRect bounding = metrics.boundingRect(elided);
//				topLeft += bounding.height();
//				painter->drawText(5,topLeft,elided);
//			}
//
//			// =========== top right
//			QStringList topRightInfo = pair.second->topRightInfo();
//			QRect topRightBounding = computeBoundingBox(metrics, topRightInfo, halfWidth);
//			painter->fillRect(rect.width()-topRightBounding.width()-10,topRight,topRightBounding.width()+10, topRightBounding.height()+5,QColor(127,127,127,127));
//			for(const QString& info : topRightInfo){
//				QString elided = metrics.elidedText(info,Qt::TextElideMode::ElideRight,halfWidth);
//				QRect bounding = metrics.boundingRect(elided);
//				int xoff = rect.width()-bounding.width()-5;
//				topRight += bounding.height();
//				painter->drawText(xoff, topRight,elided);
//			}
//
//			// ====== bottom right
//			QStringList bottomRightInfo = pair.second->bottomRightInfo();
//			QRect bottomRightBounding = computeBoundingBox(metrics, bottomRightInfo, rect.width());
//			painter->fillRect(rect.width()-bottomRightBounding.width()-10,bottomRight,bottomRightBounding.width()+10, bottomRightBounding.height()+5,QColor(127,127,127,127));
//			for(const QString& info : bottomRightInfo){
//				QString elided = metrics.elidedText(info,Qt::TextElideMode::ElideLeft,rect.width());
//				QRect bounding = metrics.boundingRect(elided);
//				int xoff = rect.width()-bounding.width()-5;
//				bottomRight -= bounding.height();
//				painter->fillRect(xoff,bottomRight,bounding.width(), bounding.height(),QColor(127,127,127,127));
//			}
//		}
	}
};

AbstractSectionView::AbstractSectionView(bool restictToMonoTypeSplit,ViewType type, 
QString uniqueName) :
Abstract2DInnerView(restictToMonoTypeSplit,new SliceQGLGraphicsView(),
				new SliceQGLGraphicsView(),
				new SliceQGLGraphicsView(), uniqueName) {
	m_viewType = type;
	if (type == ViewType::InlineView) {
		statusBar()->setWorldCoordinateLabels("Xline", "Depth");
	} else if (type == ViewType::XLineView) {
		statusBar()->setWorldCoordinateLabels("Inline", "Depth");
	}

	m_verticalAxis = nullptr;
	m_horizontalAxis = nullptr;

	m_currentSliceIJPosition=0;
	m_currentSliceWorldPosition=0;

	// Overlay for values on cursor position
	m_overlayPainter = new SyncViewer2dOverlayPainter(this);
	m_view->addOverlayPainter(m_overlayPainter);

	m_sectionType = SampleUnit::NONE;

	m_isXAxisReversed = false;

	m_orientationButton = new QPushButton;
	m_orientationButton->setStyleSheet("QPushButton {background-color: rgba(0, 0, 0, 0%); min-width: 10px; border: none;}");

	QIcon buttonIcon = getOrientationIcon();
	m_orientationButton->setIcon(buttonIcon);
	m_orientationButton->setIconSize(QSize(32, 32));
	setScenesTopCornerWidget(m_orientationButton);

	connect(m_orientationButton, &QPushButton::clicked, this, &AbstractSectionView::toggleOrientationUI);
	connect(m_scene, &QGraphicsScene::sceneRectChanged, this, &AbstractSectionView::mainSceneRectChanged);
}
void AbstractSectionView::updateSlicePosition(int worldVal, int imageVal) {
	m_currentSliceIJPosition=imageVal;
	m_currentSliceWorldPosition=worldVal;

	for (AbstractGraphicRep *r : m_visibleReps) {
		if (SliceRep *slice = dynamic_cast<SliceRep*>(r)) {
			slice->setSliceWorldPosition(worldVal);
		}
		if (ISliceableRep *slice = dynamic_cast<ISliceableRep*>(r)) {
			slice->setSliceIJPosition(imageVal);
		}
	}
}

int AbstractSectionView::getCurrentSliceWorldPosition() const {
	return m_currentSliceWorldPosition;
}


QVector3D AbstractSectionView::viewWorldTo3dWord(QPointF posi)
{
	double worldX = posi.x();
	double worldY = posi.y();
	SliceRep *rep = firstSlice();
	if (rep == nullptr)
		return QVector3D(0,0,0);

	double realX, realY;
	Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) rep->data();
	SeismicSurvey *survey = dataset->survey();
	int pos =m_currentSliceWorldPosition;
	if (m_viewType == ViewType::InlineView) {
		survey->inlineXlineToXYTransfo()->imageToWorld(worldX, pos, realX,
				realY);
	} else if (m_viewType == ViewType::XLineView) {
		survey->inlineXlineToXYTransfo()->imageToWorld(pos, worldX, realX,
				realY);
	}
	return QVector3D(realX,realY,worldY);
}


bool AbstractSectionView::absoluteWorldToViewWorld(MouseTrackingEvent &event) {
	double worldX = event.worldX();
	double worldY = event.worldY();
	SliceRep *rep = firstSlice();
	if (rep == nullptr)
		return false;

	double imageX, imageY;

	Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) rep->data();
	SeismicSurvey *survey = dataset->survey();
	int pos = m_currentSliceWorldPosition;
	int newPos = 0;
	if (m_viewType == ViewType::InlineView) {
		survey->inlineXlineToXYTransfo()->worldToImage(worldX, worldY, imageX,
				imageY);
		if (std::abs(pos - imageY) > 10)
			return false;
		newPos = imageX;
	} else if (m_viewType == ViewType::XLineView) {
		survey->inlineXlineToXYTransfo()->worldToImage(worldX, worldY, imageX,
				imageY);
		if (std::abs(pos - imageX) > 10)
			return false;
		newPos = imageY;
	}
	if (event.hasDepth())
		event.setPos(newPos, event.depth());
	else {
		double origin;
		dataset->sampleTransformation()->direct(0, origin);
		event.setPos(newPos, origin);
	}
	return true;
}
bool AbstractSectionView::viewWorldToAbsoluteWorld(MouseTrackingEvent &event) {
	double worldX = event.worldX();
	double worldY = event.worldY();
	SliceRep *rep = firstSlice();
	if (rep == nullptr)
		return false;

	double realX, realY;
	Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) rep->data();
	SeismicSurvey *survey = dataset->survey();
	int pos =m_currentSliceWorldPosition;
	if (m_viewType == ViewType::InlineView) {
		survey->inlineXlineToXYTransfo()->imageToWorld(worldX, pos, realX,
				realY);
	} else if (m_viewType == ViewType::XLineView) {
		survey->inlineXlineToXYTransfo()->imageToWorld(pos, worldX, realX,
				realY);
	}
	event.setPos(realX, realY, worldY, m_sectionType);
	return true;
}

SliceRep* AbstractSectionView::firstSlice() const {
	int count = 0;
	for (AbstractGraphicRep *r : m_visibleReps) {
		if (SliceRep *slice = dynamic_cast<SliceRep*>(r)) {
			return slice;
		}
	}
	return nullptr;
}

SliceRep* AbstractSectionView::lastSlice() const {
	int count = 0;
	for (long i=m_visibleReps.count()-1; i>=0; i--) {
		AbstractGraphicRep *r = m_visibleReps[i];
		if (SliceRep *slice = dynamic_cast<SliceRep*>(r)) {
			return slice;
		}
	}
	return nullptr;
}

void AbstractSectionView::addAxis(IGeorefImage *image) {
	m_verticalAxis = new QGLFixedAxisItem(image, VERTICAL_AXIS_SIZE, 5,
			QGLFixedAxisItem::Direction::VERTICAL);
	m_verticalAxisScene->addItem(m_verticalAxis);
	m_verticalAxisScene->setSceneRect(m_verticalAxis->boundingRect());

	m_horizontalAxis = new QGLFixedAxisItem(image, HORIZONTAL_AXIS_SIZE, 5,
			QGLFixedAxisItem::Direction::HORIZONTAL);
	m_horizontalAxisScene->addItem(m_horizontalAxis);
	m_horizontalAxisScene->setSceneRect(m_horizontalAxis->boundingRect());
	if (m_isXAxisReversed) {
		m_horizontalAxis->setTextFlip(m_isXAxisReversed);
	}

	updateVerticalAxisColor();
	updateAxisFromLengthUnit();
}

void AbstractSectionView::defineScale(SliceRep *rep) {
	Seismic3DAbstractDataset *dst = (Seismic3DAbstractDataset*) rep->data();
	double il, xl;
	double x, y;
	double x1, y1;

	dst->ijToInlineXlineTransfo()->imageToWorld(0, 0, il, xl);
	if (m_viewType == ViewType::InlineView) {
		dst->survey()->inlineXlineToXYTransfo()->imageToWorld(xl, il, x, y);
		dst->survey()->inlineXlineToXYTransfo()->imageToWorld(xl + 1, il, x1,
				y1);

		double scale = std::sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y));
		m_scaleItem->setMapScale(scale);
	} else if (m_viewType == ViewType::XLineView) {
		dst->survey()->inlineXlineToXYTransfo()->imageToWorld(xl, il, x, y);
		dst->survey()->inlineXlineToXYTransfo()->imageToWorld(xl, il + 1, x1,
				y1);

		double scale = std::sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y));
		m_scaleItem->setMapScale(scale);
	}
}

void AbstractSectionView::removeAxis() {
	if (m_verticalAxis != nullptr) {
		m_verticalAxisScene->removeItem(m_verticalAxis);
		delete m_verticalAxis;
		m_verticalAxis = nullptr;
	}
	if (m_horizontalAxis != nullptr) {
		m_horizontalAxisScene->removeItem(m_horizontalAxis);
		delete m_horizontalAxis;
		m_horizontalAxis = nullptr;
	}
}

AbstractSectionView::~AbstractSectionView() {
	m_view->removeOverlayPainter(m_overlayPainter);
	delete m_overlayPainter;
}

/**
 * Contextual menu based on Graphics item
 */
void AbstractSectionView::contextualMenuFromGraphics(double worldX, double worldY, QContextMenuEvent::Reason reason, QMenu& menu) {
//	QMenu menu("Seismic", m_view);
	m_contextualWorldX = worldX;
	m_contextualWorldY = worldY;

	QAction *showValuesAction = menu.addAction(("Show Values"));
	showValuesAction->setCheckable(m_isShowValues);
	connect(showValuesAction, SIGNAL(triggered(bool)), this,
			SLOT(showValues(bool)));

	QAction *displayGraphicTool = menu.addAction(("Display Graphic Tools"));
	connect(displayGraphicTool, SIGNAL(triggered(bool)), this,
			SLOT(startGraphicToolsDialog()));

	QAction *saveGraphicLayer = menu.addAction(("Save Graphic Layer"));
	connect(saveGraphicLayer, SIGNAL(triggered(bool)), this,
			SLOT(saveGraphicLayer()));

	QAction *loadCultural = menu.addAction(("Load/Manage Graphic Layer"));
	connect(loadCultural, SIGNAL(triggered(bool)), this,
			SLOT(loadCultural()));

//	QAction *actionSpectrum = menu.addAction(("Layer Spectrum Computation"));
//	connect(actionSpectrum, SIGNAL(triggered()), this,
//			SLOT(spectrumDecomposition()));

//	QPoint mapPos = m_view->mapFromScene(QPointF( worldX, worldY));
//	QPoint globalPos = m_view->mapToGlobal(mapPos);
//	menu.exec(globalPos);
}

/**
 * Start Layer Spectrum computation
 */
//void AbstractSectionView::spectrumDecomposition() {
//	SliceRep* datasetS = nullptr;
//	SliceRep* datasetT = nullptr;
//	double geologicalTime = 0;
//	for (AbstractGraphicRep *rep : m_visibleReps) {
//		SliceRep *slice = nullptr;
//		if ( ! (slice = dynamic_cast<SliceRep*>(rep))) {
//			continue;
//		}
//		IData* ds = slice->data();
//		QString name = ds->name();
//		Seismic3DDataset* dataset = nullptr;
//		if ( ! (dataset = dynamic_cast<Seismic3DDataset*>(ds))) {
//			continue;
//		}
//		qDebug() << "Data Name: " << name <<
//				"Data Type: " << dataset->type();
//
//		if (dataset->type() == Seismic3DAbstractDataset::CUBE_TYPE::Seismic) {
//			datasetS = slice;
//		} if (dataset->type() == Seismic3DAbstractDataset::CUBE_TYPE::RGT) {
//			datasetT = slice;
//			if (IMouseImageDataProvider *provider =
//					dynamic_cast<IMouseImageDataProvider*>(rep)) {
//				IMouseImageDataProvider::MouseInfo info;
//				if (provider->mouseData(m_contextualWorldX, m_contextualWorldY, info)) {
//					std::stringstream ss;
//					//ss << std::fixed << std::setprecision(2);
//					if (info.values.size() > 1) {
//						ss << "[";
//						for (int i = 0; i < info.values.size() - 1; i++) {
//							ss << info.values[i] << ",";
//						}
//						ss << info.values[info.values.size() - 1];
//					} else if (info.values.size() == 1) {
//						ss << info.values[0];
//					}
//					qDebug() << "Data Value: " << QString(ss.str().c_str());
//					geologicalTime = info.values[0];
//				}
//			}
//		}
//	}
//
//	//TODO
//	QPointF referencePoint(m_contextualWorldX, m_contextualWorldY);
//	if ( m_layerSpectrumDialog == nullptr ) {
//		if (datasetS!=nullptr && datasetT!=nullptr) {
//			m_layerSpectrumDialog = new LayerSpectrumDialog(
//					datasetS, datasetT, this);
//			m_layerSpectrumDialog->show();
//			//popWidget(m_layerSpectrumDialog->initWidget());
//
//
//
//			m_layerSpectrumDialog->setPoint(referencePoint);
//		}
//	}
//	else {
//		//m_layerSpectrumDialog->setGeologicalTime(tau, polarity);
//		//m_layerSpectrumDialog->updateData();
//		m_layerSpectrumDialog->setPoint(referencePoint);
//		m_layerSpectrumDialog->updateData();
//		// m_layerSpectrumDialog->trt_compute();
//	}
////	LayerSpectrumDialog* dialog = new LayerSpectrumDialog(m_survey, m_parent);
////	int result = dialog->exec();
////
////	if ( m_layerSpectrumDialog == nullptr ) {
////		if (nb > 1) {
////			m_layerSpectrumDialog = new LayerSpectrumDialog(
////					ds1, ds2, this, viewer);
//////			m_layerSpectrumDialog->show();
////			popWidget(m_layerSpectrumDialog->initWidget());
////			m_layerSpectrumDialog->setPoint(referencePoint);
////		}
////	}
////	else {
////		//m_layerSpectrumDialog->setGeologicalTime(tau, polarity);
////		//m_layerSpectrumDialog->updateData();
////		m_layerSpectrumDialog->setPoint(referencePoint);
////		m_layerSpectrumDialog->updateData();
////	}
//}


/**
 * Start show values on cursor move
 */
void AbstractSectionView::showValues(bool checked) {
	if (!m_isShowValues) {
		m_isShowValues = true;
	} else {
		m_isShowValues = false;
	}
}

bool AbstractSectionView::isMapRelationSet() const {
	SliceRep *rep = firstSlice();
	return rep != nullptr;
}

QPointF AbstractSectionView::getPointOnMap(int worldVal) const {
	QPointF out;

	if (isMapRelationSet()) {
		SliceRep *rep = firstSlice();
		if (rep == nullptr)
			return QPointF(); // should not be possible

		Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) rep->data();
		SeismicSurvey *survey = dataset->survey();

		double mapX, mapY, imageInline, imageXLine;
		if (m_viewType==ViewType::InlineView) {
			imageInline = m_currentSliceWorldPosition;
			imageXLine = worldVal;
		} else {
			imageInline = worldVal;
			imageXLine = m_currentSliceWorldPosition;
		}
		survey->inlineXlineToXYTransfo()->imageToWorld(imageXLine, imageInline, mapX, mapY);
		out = QPointF(mapX, mapY);
	}
	return out;
}

std::pair<QPointF, QPointF> AbstractSectionView::getSectionSegment() const {
	QPointF a, b;
	if (isMapRelationSet()) {
		double mapXa, mapYa, mapXb, mapYb, imageInlineA, imageXLineA, imageInlineB, imageXLineB;
		SliceRep *rep = firstSlice();
		if (rep == nullptr)
			return std::pair<QPointF, QPointF>(a, b); // should not be possible

		Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) rep->data();
		SeismicSurvey *survey = dataset->survey();

		double inlineMax, xlineMax, inlineMin, xlineMin;
		dataset->ijToInlineXlineTransfo()->imageToWorld(dataset->width()-1, dataset->height()-1, xlineMax, inlineMax);
		dataset->ijToInlineXlineTransfo()->imageToWorld(0, 0, xlineMin, inlineMin);

		QRectF rect = m_scene->sceneRect();
		if (m_viewType==ViewType::InlineView) {

			imageInlineA = m_currentSliceWorldPosition;
			imageInlineB = m_currentSliceWorldPosition;

			imageXLineA = xlineMin;
			imageXLineB = xlineMax;
		} else {
			imageInlineA = rect.left();
			imageInlineB = rect.right();
			imageXLineA = inlineMin;
			imageXLineB = inlineMax;
		}

		survey->inlineXlineToXYTransfo()->imageToWorld(imageXLineA, imageInlineA, mapXa, mapYa);
		survey->inlineXlineToXYTransfo()->imageToWorld(imageXLineB, imageInlineB, mapXb, mapYb);

		a = QPointF(mapXa, mapYa);
		b = QPointF(mapXb, mapYb);
	}

	return std::pair<QPointF, QPointF>(a, b);
}

const Affine2DTransformation* AbstractSectionView::inlineXLineToXY() const {
	SliceRep *rep = firstSlice();
	if (rep!=nullptr) {
		Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) rep->data();
		SeismicSurvey *survey = dataset->survey();
		return survey->inlineXlineToXYTransfo();
	} else {
		return nullptr;
	}
}

double AbstractSectionView::displayDistance() const {
	return m_displayDistance;
}

void AbstractSectionView::setDisplayDistance(double val) {
	if (m_displayDistance!=val) {
		m_displayDistance = val;
		emit displayDistanceChanged(m_displayDistance);
	}
}

void AbstractSectionView::toggleXAxis(bool toggle) {
	if (toggle!=m_isXAxisReversed) {
		m_isXAxisReversed = toggle;
		m_view->scale(-1, 1);
		m_horizontalAxisView->scale(-1, 1);
		if (m_horizontalAxis) {
			m_horizontalAxis->setTextFlip(m_isXAxisReversed);
		}
	}
}

void AbstractSectionView::toggleOrientationUI() {
	toggleXAxis(!m_isXAxisReversed);
	QIcon buttonIcon = getOrientationIcon();
	m_orientationButton->setIcon(buttonIcon);
}

QIcon AbstractSectionView::getOrientationIcon() {
	if (m_isXAxisReversed) {
		return QIcon(":/slicer/icons/Flipsection.svg");
	} else {
		return QIcon(":/slicer/icons/Flipsectioninversé.svg");
	}
}

void AbstractSectionView::updateTitleFromSlices() {
	if (m_suffixTitle.isEmpty() || m_suffixTitle.isNull()) {
		// no name, search for it
		QString rgtName;
		SliceRep* sliceRgtObj = nullptr;
		QString name;
		SliceRep* sliceObj = nullptr;
		int i = 0;
		while ((name.isNull() || name.isEmpty()) && i<m_visibleReps.size()) {
			AbstractGraphicRep *r = m_visibleReps[i];
			SliceRep *slice = dynamic_cast<SliceRep*>(r);
			if (slice && slice->data()) {
				Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(slice->data());
				if (dataset && dataset->type()!=Seismic3DAbstractDataset::RGT) {
					name = dataset->name();
					sliceObj = slice;
				} else if (dataset && (rgtName.isNull() || rgtName.isEmpty())) {
					rgtName = dataset->name();
					sliceRgtObj = slice;
				}
			}
			i++;
		}
		if ((name.isNull() || name.isEmpty()) && !rgtName.isNull() && !rgtName.isEmpty()) {
			name = rgtName;
			m_cacheIsNameRgt = true;
			m_cacheSliceNameObj = sliceRgtObj;
			updateTile(name);
		} else if (!name.isNull() && !name.isEmpty()) {
			m_cacheIsNameRgt = false;
			m_cacheSliceNameObj = sliceObj;
			updateTile(name);
		} else {
			updateTile("");
		}
	} else if (!m_cacheIsNameRgt) {
		// search if name holder is still there
		bool nameHolderPresent = false;
		int i = 0;
		while (!nameHolderPresent && i<m_visibleReps.size()) {
			nameHolderPresent = m_visibleReps[i]==m_cacheSliceNameObj;
			i++;
		}
		if (!nameHolderPresent) {
			updateTile("");
			updateTitleFromSlices();
		}
	} else {
		bool nameHolderPresent = false;
		bool foundSeismic = false;
		int i = 0;
		while ((!nameHolderPresent || !foundSeismic) && i<m_visibleReps.size()) {
			if (!nameHolderPresent) {
				nameHolderPresent = m_visibleReps[i]==m_cacheSliceNameObj;
			}
			if (!foundSeismic) {
				SliceRep* slice = dynamic_cast<SliceRep*>(m_visibleReps[i]);
				bool valid = slice!=nullptr;
				if (valid) {
					Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(slice->data());
					valid = dataset->type()!=Seismic3DAbstractDataset::RGT;
				}
				foundSeismic = valid;
			}
			i++;
		}
		if (!nameHolderPresent || foundSeismic) {
			updateTile("");
			updateTitleFromSlices();
		}
	}
}

void AbstractSectionView::setDepthLengthUnitProtected(const MtLengthUnit* depthLengthUnit) {
	Abstract2DInnerView::setDepthLengthUnitProtected(depthLengthUnit);

	updateAxisFromLengthUnit();
}

void AbstractSectionView::updateAxisFromLengthUnit() {
	double a;
	if (m_sectionType==SampleUnit::DEPTH) {
		a = MtLengthUnit::convert(MtLengthUnit::METRE, *m_depthLengthUnit, 1);
	} else {
		a = 1;
	}
	AffineTransformation transfo(a, 0);
	if (m_verticalAxis) {
		m_verticalAxis->setDisplayValueTransform(&transfo);
		m_verticalAxisScene->update(m_verticalAxisScene->sceneRect());
	}
}

void AbstractSectionView::mainSceneRectChanged() {
	if (m_verticalAxisScene) {
		QRectF sceneRect = m_verticalAxisScene->sceneRect();
		sceneRect.setY(m_scene->sceneRect().y());
		sceneRect.setHeight(m_scene->sceneRect().height());
		m_verticalAxisScene->setSceneRect(sceneRect);
	}
	if (m_horizontalAxisScene) {
		QRectF sceneRect = m_horizontalAxisScene->sceneRect();
		sceneRect.setX(m_scene->sceneRect().x());
		sceneRect.setWidth(m_scene->sceneRect().width());
		m_horizontalAxisScene->setSceneRect(sceneRect);
	}
}

void AbstractSectionView::updateVerticalAxisColor() {
	if (m_verticalAxis) {
		switch (m_sectionType) {
		case SampleUnit::DEPTH:
			m_verticalAxis->setColor(Qt::red);
			break;
		case SampleUnit::TIME:
			m_verticalAxis->setColor(QColor::fromRgb(33, 170 , 255));
			break;
		default:
			m_verticalAxis->resetColor();
			break;
		}
	}
}

double AbstractSectionView::getWellSectionWidth() const {
	return m_wellSectionWidth;
}

void AbstractSectionView::setWellSectionWidth(double value) {
	if(m_wellSectionWidth != value) {
		m_wellSectionWidth = value;
		emit signalWellSectionWidth(m_wellSectionWidth);
	}
}
