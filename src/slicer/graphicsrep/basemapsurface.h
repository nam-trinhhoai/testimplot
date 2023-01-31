#ifndef SRC_SLICER_GRAPHICSREP_BASEMAPSURFACE_H
#define SRC_SLICER_GRAPHICSREP_BASEMAPSURFACE_H

#include "viewutils.h"

#include <QObject>

class QGraphicsObject;
class IImagePaletteHolder;

class BaseMapSurface : public QObject {
public:
	/**
	 * Take ownership of basemapItem and isoSurface
	 *
	 * isoSurface contains the iso in time/depth not in pixel
	 */
	BaseMapSurface(QGraphicsObject* basemapItem, IImagePaletteHolder* isoSurface, SampleUnit isoType, QObject* parent=nullptr);
	~BaseMapSurface();

	QGraphicsObject* basemapItem();
	IImagePaletteHolder* isoSurface();
	SampleUnit isoType() const;

	bool updateImages(BaseMapSurface* surface);

private:
	// only to be called by m_basemapItem destroyed signal
	void resetBaseMapItem();

	QGraphicsObject* m_basemapItem = nullptr;
	IImagePaletteHolder* m_isoSurface = nullptr;
	SampleUnit m_isoType = SampleUnit::NONE;
};

#endif
