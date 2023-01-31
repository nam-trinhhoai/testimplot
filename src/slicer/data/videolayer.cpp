#include "videolayer.h"
#include "videolayergraphicrepfactory.h"
#include "seismic3dabstractdataset.h"

#include <QFileInfo>

VideoLayer::VideoLayer(WorkingSetManager * workingSet, QString idPath,
		Seismic3DAbstractDataset* originDataset, QObject *parent) :
		IData(workingSet, parent), IFileBasedData(idPath) {
	m_originDataset = originDataset;
	m_mediaPath = idPath;

	m_name = QFileInfo(idPath).baseName();
	m_width = m_originDataset->width();
	m_height = m_originDataset->depth();

	m_uuid = QUuid::createUuid();

	m_repFactory = new VideoLayerGraphicRepFactory(this);
}

VideoLayer::~VideoLayer() {
	delete m_repFactory;
}

const Affine2DTransformation  * const VideoLayer::ijToXYTransfo() const {
	m_originDataset->ijToXYTransfo();
}

IGraphicRepFactory *VideoLayer::graphicRepFactory() {
	return m_repFactory;
}

QUuid VideoLayer::dataID() const {
	return m_uuid;
}
