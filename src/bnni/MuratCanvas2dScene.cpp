#include "rectanglemovable.h"
#include "SelectionRectangle.h"

#include <QVector>
#include <QPixmap>
#include <QDebug>
#include <QEvent>
#include <QGraphicsLineItem>
#include <QGraphicsPathItem>
#include <QGraphicsPixmapItem>
#include <QGraphicsTextItem>
#include <QGraphicsPolygonItem>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneWheelEvent>
#include <QKeyEvent>
#include <QGraphicsView>
#include <QColor>
#include <cmath>
#include "MuratCanvas2dScene.h"
#include <cfloat>
#include <vector>
#include <utility>
#include "sampletypebinder.h"
#include "cutline.h"

#include <QGraphicsPolygonItem>
//#include "EditablePolygonItem.h"

//using namespace io;

/*
 * Public Constructor of VisualScene class
 *
 * QString filename : Path of Main Image
 *
 * Create Image
 * Create Cursor and Curves (empty)
 * Init ratio
 */
MuratCanvas2dScene::MuratCanvas2dScene(QObject *parent) :
		QGraphicsScene(parent) {
	// Instanciate image
	picture = nullptr;
	isImageMode = true;

	// Create default QPen and default QColor
	pen = new QPen();
	pen->setCosmetic(true);
	pen->setWidth(1);
	hColor = new QColor(0, 255, 0);
    vColor = new QColor(0, 0, 0);

	// Create cursor
	hCursor = this->addLine(QLineF(), hQPen());
	vCursor = this->addLine(QLineF(), vQPen());

	hCursor->setZValue(1000000);
	vCursor->setZValue(1000000);

	hCursor->hide();
	vCursor->hide();

	// Create curves to plot
	hCurve = this->addPath(QPainterPath(), hQPen());
	vCurve = this->addPath(QPainterPath(), vQPen());

	hCurve->setZValue(1000000);
	vCurve->setZValue(1000000);

	hCurve->hide();
	vCurve->hide();

	patchSize = 0;
    patch = new RectangleMovable(false, 1, 1, this);
	this->addItem(patch);
	patch->setRectHeight(patchSize);
	patch->setRectWidth(patchSize);
	patch->setZValue(1000000);

	// Init scale ratio
    ratioX = 1;
    ratioY = 1;

	displayValue = new QGraphicsTextItem("0");
	displayValue->setFlag(QGraphicsItem::ItemIgnoresTransformations);
	displayValue->setDefaultTextColor(QColor::fromRgb(255, 215, 0));
	this->addItem(displayValue);
	displayValue->hide();
	displayValue->setZValue(1000000);//cursor over everything

	curveOffset = -150;
	curveSize = 150;
}

/*
 * Public Destructor
 *
 * Delete all pointers
 */
MuratCanvas2dScene::~MuratCanvas2dScene() {
	for(auto ext:extensions){
		ext->releaseCanvas(this);
	}

	if (picture) {
		delete picture;
	}
	delete hCursor;
	delete vCursor;
	delete hCurve;
	delete vCurve;
	delete pen;
	delete hColor;
	delete vColor;

	delete patch;
	delete displayValue;
}

bool MuratCanvas2dScene::getIsImageMode() {
	return isImageMode;
}

void MuratCanvas2dScene::setIsImageMode(bool val) {
	isImageMode = val;
}

/*
 * Public Class Method
 *
 * Update scale of all items to make them visible after scaling
 */
void MuratCanvas2dScene::updateCursor(qreal sx, qreal sy) {
    ratioX /= sx;
    ratioY /= sy;

    patch->updateSize(sx, sy);
	QGraphicsView* _parent = ((QGraphicsView*) this->parent());

	for (RectangleMovable* item : overlayRectangles) {
        item->updateSize(sx, sy);
	}

    for (int i=0; i<lines.size(); i++) {
        if (lines[i]!=nullptr) {
            lines[i]->updateSize(sx, sy);
        }
    }
}

/*
 * Public Class method
 *
 * Return Image size
 */
QSize MuratCanvas2dScene::imageSize() {
	QSize size;
	if (picture) {
		QImage img = picture->pixmap().toImage();
		size = img.size();
	}

	return size;
}

/*
 * Public Class method
 *
 * Return Image Rectangle
 */
QRect MuratCanvas2dScene::imageRect() const {
	QRect rectangle;
	if (picture) {
		rectangle = picture->pixmap().toImage().rect();
	}
	return rectangle;
}

/*
 * Public Class method
 *
 * Set main image
 */
void MuratCanvas2dScene::setImage(const QImage image, Matrix2DInterface* mat) {
	if (picture == nullptr) {
        cache_image = image;
		picture = this->addPixmap(QPixmap::fromImage(image));
		picture->setShapeMode(QGraphicsPixmapItem::BoundingRectShape);
	} else {
		picture->setPixmap(QPixmap::fromImage(image));

	}
	img = image;
	picture->setZValue(0);
	this->mat = mat;
	displayValue->show();
	if (image.width() == 1) {
		setIsImageMode(false);
	} else {
		setIsImageMode(true);
	}
	updateCurvesAndCursor(QPointF(0, 0));

	for (int i=0; i<lines.size(); i++) {
	    if (lines[i]!=nullptr) {
	        lines[i]->setMax(mat->height()-1);
	    }
	}
}

/*
 * Public Class method
 *
 * Return image Pixmap
 */
const QPixmap MuratCanvas2dScene::getImage() {
	if (picture) {
		return picture->pixmap();
	} else {
		return QPixmap();
	}
}

/*
 * Public Class method
 *
 * Return image Pixmap
 */
const Matrix2DInterface* MuratCanvas2dScene::getMatrix() {
	return mat;
}

void MuratCanvas2dScene::toggleCurve(bool val) {
    showCurve = val;
    if (!showCurve) {
        setHorizontalCurve(false);
        setVerticalCurve(false);
    }
}

/*
 * Public method
 *
 * Setter of activation of cursor vertical
 */
void MuratCanvas2dScene::setVerticalCursor(bool b) {
	verticalCursor = b;
	if (verticalCursor) {
		vCursor->show();
	} else {
		vCursor->hide();
	}
}

/*
 * Public method
 *
 * Getter of activation of cursor vertical
 */
bool MuratCanvas2dScene::getVerticalCursor() {
	return verticalCursor;
}

/*
 * Public method
 *
 * Setter of activation of cursor horizontal
 */
void MuratCanvas2dScene::setHorizontalCursor(bool b) {
	horizontalCursor = b;
	if (horizontalCursor) {
		hCursor->show();
	} else {
		hCursor->hide();
	}
}

/*
 * Public method
 *
 * Setter of activation of cursor vertical
 */
void MuratCanvas2dScene::setVerticalCurve(bool b) {
	verticalCurve = b;
	if (verticalCurve) {
		vCurve->show();
	} else {
		vCurve->hide();
	}
}

/*
 * Public method
 *
 * Getter of activation of cursor vertical
 */
bool MuratCanvas2dScene::getVerticalCurve() {
	return verticalCurve;
}

/*
 * Public method
 *
 * Setter of activation of cursor horizontal
 */
void MuratCanvas2dScene::setHorizontalCurve(bool b) {
	horizontalCurve = b;
	if (horizontalCurve) {
		hCurve->show();
	} else {
		hCurve->hide();
	}
}

/*
 * Public method
 *
 * Getter of activation of cursor horizontal
 */
bool MuratCanvas2dScene::getHorizontalCurve() {
	return horizontalCurve;
}

/**
 * Add an extension to the canvas.
 */
void MuratCanvas2dScene::addExtension(Canvas2DExtension* extension) {
	auto it = std::find(extensions.begin(), extensions.end(),extension);
	if(it == extensions.end()){
		extensions.push_back(extension);
	}
}

/**
 * Remove an extension from the canvas.
 */
void MuratCanvas2dScene::removeExtension(Canvas2DExtension* extension) {
	auto it = std::find(extensions.begin(), extensions.end(),extension);
	if(it != extensions.end()){
		extensions.erase(it);
	}
}



/*
 * Protected Class Method
 *
 * Trigger update Cursor position and Curves
 */
void MuratCanvas2dScene::simulateMousePressEvent(QPointF pos) {
	bool tmpHorizontalCurve = horizontalCurve;
	bool tmpVerticalCurve = verticalCurve;

    setHorizontalCurve(showCurve);
    setVerticalCurve(showCurve);

	updateCurvesAndCursor(pos);
	horizontalCurve = tmpHorizontalCurve;
	verticalCurve = tmpVerticalCurve;

}

/*
 * Protected Class Method
 *
 * Trigger update Cursor position and Curves
 */
void MuratCanvas2dScene::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *ev) {
	for (auto ext : extensions) {
		if (ext->isEnabled())
			ext->mouseDoubleClickEvent(ev);
		if (ev->isAccepted())
			return;
	}
	QGraphicsScene::mouseDoubleClickEvent(ev);
}

void MuratCanvas2dScene::mouseMoveEvent(QGraphicsSceneMouseEvent *ev) {
	for (auto ext : extensions) {
		if (ext->isEnabled())
			ext->mouseMoveEvent(ev);
		if (ev->isAccepted())
			return;
	}
	this->QGraphicsScene::mouseMoveEvent(ev);

	updateCurvesAndCursor(ev->scenePos());
}

void MuratCanvas2dScene::mousePressEvent(QGraphicsSceneMouseEvent *ev) {
	for (auto ext : extensions) {
		if (ext->isEnabled())
			ext->mousePressEvent(ev);
		if (ev->isAccepted())
			return;
	}

	this->QGraphicsScene::mousePressEvent(ev);

	if (ev->buttons() == Qt::MiddleButton) {
		return;
	}

	simulateMousePressEvent(ev->scenePos());

	int x = patch->pos().x() + patchSize / 2;
	int y = patch->pos().y() + patchSize / 2;
	emit takePatch(x, y);
}

void MuratCanvas2dScene::mouseReleaseEvent(QGraphicsSceneMouseEvent *ev) {
	for (auto ext : extensions) {
		if (ext->isEnabled())
			ext->mouseReleaseEvent(ev);
		if (ev->isAccepted())
			break;
	}
	QGraphicsScene::mouseReleaseEvent(ev);
}

void MuratCanvas2dScene::wheelEvent(QGraphicsSceneWheelEvent *ev) {
	for (auto ext : extensions) {
		if (ext->isEnabled())
			ext->wheelEvent(ev);
		if (ev->isAccepted())
			return;
	}
	QGraphicsScene::wheelEvent(ev);
}

/**
 * Keyboard binding
 */
void MuratCanvas2dScene::keyPressEvent(QKeyEvent *ev) {
	ev->setAccepted(false);
	for (auto ext : extensions) {
		if (ext->isEnabled())
			ext->keyPressEvent(ev);
		if (ev->isAccepted())
			return;
	}
	QGraphicsScene::keyPressEvent(ev);
}
void MuratCanvas2dScene::keyReleaseEvent(QKeyEvent *ev) {
	ev->setAccepted(false);
	for (auto ext : extensions) {
		if (ext->isEnabled())
			ext->keyReleaseEvent(ev);
		if (ev->isAccepted())
			return;
	}
	QGraphicsScene::keyReleaseEvent(ev);
}

/*
 * Protected Class Method
 *
 * Update lines from cursor and curves of gray signal
 */
void MuratCanvas2dScene::updateCurvesAndCursor(QPointF scenePos) {

	setHorizontalCurve(horizontalCurve);
	setVerticalCurve(verticalCurve);
	if (isImageMode) {
		updateCurvesAndCursorImage(scenePos);
	} else {
		updateCurvesAndCursorLog(scenePos);
	}

}

template<typename InputType>
struct ReadHorizontalLine {
	static void run(Matrix2DInterface* mat, QVector<unsigned int>& histo,
			int step, QPolygonF& hPoly, qreal minMat, qreal maxMat, int y) {
		InputType* matrix = (InputType*) mat->data();

		int w = mat->width();
		for (int i = 0; i < w; i += step) {
			double hGray = matrix[i + w * y];
			hPoly << QPoint(i, (int) std::floor(hGray));
			int index = std::floor(
					(std::max(minMat, std::min(hGray, maxMat)) - minMat) * 256
							/ (maxMat - minMat));
			if (index == 256) {
				index = 255;
			}
			histo[index]++;
		}
	}
};

template<typename InputType>
struct ReadVerticalLine {
	static void run(Matrix2DInterface* mat, QVector<unsigned int>& histo,
			int step, QPolygonF& vPoly, qreal minMat, qreal maxMat, int x) {
		InputType* matrix = (InputType*) mat->data();
		int w = mat->width();
		int h = mat->height();

		for (int i = 0; i < h; i += step) {
			double vGray = matrix[x + w * i];
			vPoly << QPoint((int) std::floor(vGray), i);
			int index = std::floor(
					(std::max(minMat, std::min(vGray, maxMat)) - minMat) * 256
							/ (maxMat - minMat));
			if (index == 256) {
				index = 255;
			}
			histo[index]++;
		}
	}
};

void MuratCanvas2dScene::updateCurvesAndCursorImage(QPointF scenePos) {
	if (picture == nullptr) {
		return;
	}
	//qDebug() << "Event type = " << ev->type();

	int x = scenePos.x();
	int y = scenePos.y();
	if (picture) {
		QPointF posOnItem = picture->mapFromScene(scenePos);
		x = posOnItem.x();
		y = posOnItem.y();
	}
	//qDebug() << "pressed X= " << x << ", Y= " << y;

	// Get tmp image to access pixels directly
	int h = img.height();
	int w = img.width();

	if (x < 0 || x >= w || y < 0 || y >= h) {
		//qDebug() << "Out Of Bound";
		return;
	}

	// Update Cursor
	hCursor->setLine(0, scenePos.y(), w, scenePos.y());
	vCursor->setLine(scenePos.x(), 0, scenePos.x(), h);

	// Extract gray values
	if (mat != nullptr) {
		if (minMat == maxMat) {
			maxMat = minMat + 1;
		}

		if (this->horizontalCurve) {
			QPolygonF hPoly;
			int index = 0;
			QVector<unsigned int> histo(256, 0);
			int step = 1;    //std::max(1.0, floor(ratio));

			SampleTypeBinder binder(mat->getType());
			binder.bind < ReadHorizontalLine
					> (mat, histo, step, hPoly, minMat, maxMat, y);

			if (picture) {
				hPoly = picture->mapToScene(hPoly);
			}

			index = -1;
			unsigned int sum = 0;
			while (sum < w / 2) {
				index++;
				sum += histo[index];
			}
			qreal hMedian = index * (maxMat - minMat) / 256 + minMat;

			// Create curves
			QPainterPath hQPainter = QPainterPath();
			hQPainter.addPolygon(hPoly);

			// modify position of curves
			QTransform hTransform;
			int hCurveMediaOffset = curveSize * (hMedian - minMat)
					/ (maxMat - minMat);
			hTransform.translate(0,
                    scenePos.y() + (curveOffset + hCurveMediaOffset) * getRatio() / 2);
            hTransform.scale(1, -curveSize * getRatio() / 2.0 / (maxMat - minMat));
			hTransform.translate(0, -minMat);
			hQPainter = hTransform.map(hQPainter);

			// Apply Curves
			hCurve->setPath(hQPainter);
		}

		if (this->verticalCurve) {
			QPolygonF vPoly;
			QVector<unsigned int> histo(256, 0);
			int index = 0;
			int step = 1;

			SampleTypeBinder binder(mat->getType());
			binder.bind < ReadVerticalLine
					> (mat, histo, step, vPoly, minMat, maxMat, x);

			if (picture) {
				vPoly = picture->mapToScene(vPoly);
			}

			index = -1;
			unsigned int sum = 0;
			while (sum < h / 2) {
				index++;
				sum += histo[index];
			}
			qreal vMedian = index * (maxMat - minMat) / 256 + minMat;

			// Create curves
			QPainterPath vQPainter = QPainterPath();
			vQPainter.addPolygon(vPoly);

			QTransform vTransform;
			int vCurveMediaOffset = curveSize * (vMedian - minMat)
					/ (maxMat - minMat);
			vTransform.translate(
                    scenePos.x() + (curveOffset - vCurveMediaOffset) * getRatio() / 2, 0);
            vTransform.scale(curveSize * getRatio() / 2 / (maxMat - minMat), 1);
			vTransform.translate(-minMat, 0);
			vQPainter = vTransform.map(vQPainter);

			// Apply Curves
			vCurve->setPath(vQPainter);
		}
	}

	x = (x - patchSize / 2 < 0) ? patchSize / 2 : x;
	y = (y - patchSize / 2 < 0) ? patchSize / 2 : y;
	x = (x - patchSize / 2 + patchSize > w) ?
			w - (patchSize - patchSize / 2) : x;
	y = (y - patchSize / 2 + patchSize > h) ?
			h - (patchSize - patchSize / 2) : y;
	patch->setPos(x - patchSize / 2, y - patchSize / 2);

	if (mat && verticalCursor && horizontalCursor) {

		QString string = QString("Value %1").arg(mat->getDouble(x, y)); //Update the cursor potion text

		displayValue->setPlainText(string);
        displayValue->setPos(scenePos.x() + 10 * getRatio(), scenePos.y() - 30 * getRatio());
		displayValue->show();
	} else {
		displayValue->hide();
	}
}

void MuratCanvas2dScene::updateCurvesAndCursorLog(QPointF scenePos) {
	if (picture == nullptr || mat == nullptr) {
		return;
	}
	int step = 1;
	QPolygon vPoly;
	for (int i = 0; i < picture->pixmap().height(); i += step) {
		double hGray = mat->getDouble(0, i);
        vPoly << QPoint(std::floor((std::max(minMat, std::min(hGray, maxMat))- minMat) * 256 / (maxMat - minMat)),i);
	}
	QPainterPath vQPainter = QPainterPath();
	vQPainter.addPolygon(vPoly);
	vCurve->setPath(vQPainter);
    QPen pen = vCurve->pen();
    pen.setColor(QColor(255, 255, 255));
    vCurve->setPen(pen);
	vCurve->show();
	hCursor->setLine(0, scenePos.y(), 256, scenePos.y());

}

/*
 * Public Class Method
 *
 * Update lines from cursor and curves of gray signal
 */
void MuratCanvas2dScene::simulateUpdateCurvesAndCursor(QPointF scenePos) {
	updateCurvesAndCursor(scenePos);
}

/*
 * Private Class Method
 *
 * Convenience method to get correct QPen with color and default QPen
 */
QPen MuratCanvas2dScene::buildQPenWithColor(QPen* pen, QColor* color) {
	// Create QPen using default QPen pen
	QPen result(*pen);

	// Add color
	result.setColor(*color);

	// Return new QPen
	return result;
}

/*
 * Private Class Method
 *
 * Convenience method to get correct QPen for horizontal items
 */
QPen MuratCanvas2dScene::hQPen() {
	return buildQPenWithColor(pen, hColor);
}

/*
 * Protected Class Method
 *
 * Convenience method to get correct QPen for vertical items
 */
QPen MuratCanvas2dScene::vQPen() {
	return buildQPenWithColor(pen, vColor);
}

void MuratCanvas2dScene::setPatchSize(int ps) {
	patchSize = ps;
	patch->setRectHeight(patchSize);
	patch->setRectWidth(patchSize);
}

int MuratCanvas2dScene::getPatchSize() {
	return patchSize;
}

int MuratCanvas2dScene::getCurveOffset() {
	return curveOffset;
}

void MuratCanvas2dScene::setCurveOffset(int value) {
	curveOffset = value;
}

int MuratCanvas2dScene::getCurveSize() {
	return curveSize;
}

void MuratCanvas2dScene::setCurveSize(int value) {
	curveSize = value;
}

void MuratCanvas2dScene::setMinMat(qreal val) {
	minMat = val;
}
qreal MuratCanvas2dScene::getMinMat() {
	return minMat;
}

void MuratCanvas2dScene::setMaxMat(qreal val) {
	maxMat = val;
}

qreal MuratCanvas2dScene::getMaxMat() {
	return maxMat;
}

QImage MuratCanvas2dScene::getQImage() {
	return img;
}

void MuratCanvas2dScene::drawRectangles(
		std::vector<std::pair<QRect, QColor>> rects) {
	for (std::pair<QRect, QColor> pair : rects) {
        RectangleMovable* item = new RectangleMovable(false, ratioX, ratioY, this);
		item->setColor(pair.second);
		item->setRectHeight(pair.first.height());
		item->setRectWidth(pair.first.width());
		item->setZValue(4);
		this->addItem(item);
		item->setPos(pair.first.topLeft());
		item->show();

		overlayRectangles.push_back(item);
	}
}

std::vector<std::pair<QRect, QColor>> MuratCanvas2dScene::getOverlayRectangles() {
	std::vector<std::pair<QRect, QColor>> out;
	for (RectangleMovable* item : overlayRectangles) {
		QRect rect = QRect(item->pos().x(), item->pos().y(),
				item->getRectWidth(), item->getRectHeight());
		QColor color = item->getColor();
		out.push_back(std::pair<QRect, QColor>(rect, color));
	}
	return out;
}

void MuratCanvas2dScene::clearOverlayRectangles() {
	while (overlayRectangles.size() != 0) {
		RectangleMovable* item = overlayRectangles.back();
		item->hide();
		overlayRectangles.pop_back();
		this->removeItem(item);
		item->deleteLater();
	}
}

std::vector<double> MuratCanvas2dScene::getLines() {
	std::vector<double> posLines;
	posLines.resize(lines.size(), 0);
	for (int i=0; i<lines.size(); i++) {
		if (lines[i]!=nullptr) {
			posLines[i] = lines[i]->y();
		}
	}
    return posLines;
}

void MuratCanvas2dScene::setLines(const std::vector<double>& posLines) {
    if (lines.size()>posLines.size()) {
        // Too many lines
        for (int i=lines.size()-1; i>=posLines.size(); i--) {
            CutLine* line = lines[i];
            std::vector<CutLine*>::iterator it = lines.begin();
            std::advance(it, i);
            lines.erase(it);

            if (line!=nullptr) {
                disconnect(line, SIGNAL(lineChanged(CutLine*)), this, SLOT(lineChanged(CutLine*)));
                removeItem(line);
                delete line;
            }
        }
    } else if (lines.size()<posLines.size()) {
        // Not enough lines
        int oriSize = lines.size();
        lines.resize(posLines.size(), nullptr);

        for (int i=oriSize; i<posLines.size(); i++) {
            CutLine* line = new CutLine(150, ratioX, ratioY, this);
            line->setMin(0);
            if (mat!=nullptr) {
                line->setMax(mat->height()-1);
            }
            addItem(line);
            line->show();
            lines[i] = line;
            connect(line, SIGNAL(lineChanged(CutLine*)), this, SLOT(lineChanged(CutLine*)));
        }
    }
    for (int i=0; i<lines.size(); i++) {
        if (lines[i]!=nullptr) {
            lines[i]->setPos(lines[i]->x(), posLines[i]);
        }
    }
}

void MuratCanvas2dScene::lineChanged(CutLine* line) {
    emit linesChanged(getLines());
}

qreal MuratCanvas2dScene::getRatio() {
    return ratioX;
}


void MuratCanvas2dScene::setImageItemVerticalScale(float value) {
	if (!picture) {
		return;
	}

	QTransform transform = QTransform::fromScale(1, value);

	picture->setTransform(transform);
}
