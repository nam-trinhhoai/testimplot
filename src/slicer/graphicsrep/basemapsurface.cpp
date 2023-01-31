#include "basemapsurface.h"
#include "iimagepaletteholder.h"
#include "sampletypebinder.h"
#include "CUDAImageMask.h"
#include "rgbqglcudaimageitem.h"

#include <QGraphicsObject>


BaseMapSurface::BaseMapSurface(QGraphicsObject* basemapItem, IImagePaletteHolder* isoSurface, SampleUnit isoType, QObject* parent) :
			QObject(parent) {
	m_basemapItem = basemapItem;
	m_isoSurface = isoSurface;
	m_isoType = isoType;

	connect(m_basemapItem, &QGraphicsObject::destroyed, this, &BaseMapSurface::resetBaseMapItem);
}

BaseMapSurface::~BaseMapSurface() {
	m_isoSurface->deleteLater();

	if (m_basemapItem!=nullptr) {
		disconnect(m_basemapItem, &QGraphicsObject::destroyed, this, &BaseMapSurface::resetBaseMapItem);
		m_basemapItem->deleteLater();
	}
}

QGraphicsObject* BaseMapSurface::basemapItem() {
	return m_basemapItem;
}

IImagePaletteHolder* BaseMapSurface::isoSurface() {
	return m_isoSurface;
}

SampleUnit BaseMapSurface::isoType() const {
	return m_isoType;
}

void BaseMapSurface::resetBaseMapItem() {
	disconnect(m_basemapItem, &QGraphicsObject::destroyed, this, &BaseMapSurface::resetBaseMapItem);
	m_basemapItem = nullptr;
}

template<typename DataTypeA>
struct ModifyIsoSurfaceKernelLevel1 {
	template<typename DataTypeB>
	struct ModifyIsoSurfaceKernelLevel2 {
		static void run(IImagePaletteHolder* mainImage, IImagePaletteHolder* secondImage,
				const QByteArray& mainMask, const QByteArray& secondMask) {
			unsigned int width = mainImage->width();
			unsigned int height = mainImage->height();

			secondImage->lockPointer();
			mainImage->lockPointer();

			DataTypeA* mainTab = static_cast<DataTypeA*>(secondImage->backingPointer());
			DataTypeB* secondTab = static_cast<DataTypeB*>(secondImage->backingPointer());

			for (unsigned int j=0; j< height; j++)
			{
				for (unsigned int i =0; i<width; i++)
				{
					if ((mainMask[j * width + i] == 0) &&
							(secondMask[j * width + i] == 255) )
					{
						double value = secondTab[i + width * j];
						if (value < std::numeric_limits<DataTypeA>::min()) {
							value = std::numeric_limits<DataTypeA>::min();
						}
						if (value > std::numeric_limits<DataTypeA>::max()) {
							value = std::numeric_limits<DataTypeA>::max();
						}
						mainTab[i + width * j] = value;
					}
				}
			}
			mainImage->unlockPointer();
			secondImage->unlockPointer();
		}
	};

	static void run(IImagePaletteHolder* mainImage, IImagePaletteHolder* secondImage,
			const QByteArray& mainMask, const QByteArray& secondMask) {
		SampleTypeBinder binder(secondImage->sampleType());
		binder.bind<ModifyIsoSurfaceKernelLevel2>(mainImage, secondImage, mainMask, secondMask);
	}
};

bool BaseMapSurface::updateImages(BaseMapSurface* surface) {
	bool ok = m_isoSurface->height()==surface->m_isoSurface->height() &&
			m_isoSurface->width()==surface->m_isoSurface->width() &&
			m_isoType==surface->m_isoType && dynamic_cast<CUDAImageMask*>(m_basemapItem)!=nullptr &&
			dynamic_cast<CUDAImageMask*>(surface->m_basemapItem)!=nullptr;

	if (ok) {
		CUDAImageMask* item = dynamic_cast<CUDAImageMask*>(m_basemapItem);
		CUDAImageMask* otherItem = dynamic_cast<CUDAImageMask*>(surface->m_basemapItem);
		QByteArray mask = item->getArray();
		QByteArray otherMask = otherItem->getArray();

		SampleTypeBinder binder(m_isoSurface->sampleType());
		binder.bind<ModifyIsoSurfaceKernelLevel1>(m_isoSurface, surface->m_isoSurface, mask, otherMask);

		if (RGBQGLCUDAImageItem* rgbItem = dynamic_cast<RGBQGLCUDAImageItem*>(m_basemapItem)) {
			rgbItem->updateImage(surface->m_basemapItem, m_basemapItem->parentItem());
		}

		emit m_isoSurface->dataChanged();
	}

	return ok;
}
