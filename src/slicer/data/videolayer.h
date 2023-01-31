#ifndef VideoLayer_H
#define VideoLayer_H
#include <QObject>

#include "idata.h"
#include "igeorefimage.h"
#include "ifilebaseddata.h"

#include "affine2dtransformation.h"

class Seismic3DAbstractDataset;
class VideoLayerGraphicRepFactory;

class VideoLayer: public IData, public IFileBasedData {
Q_OBJECT
public:
	VideoLayer(WorkingSetManager * workingSet, QString idPath, Seismic3DAbstractDataset* originDataset, QObject *parent =
			0);
	virtual ~VideoLayer();

	const Affine2DTransformation  * const ijToXYTransfo() const;

	//IData
	virtual IGraphicRepFactory *graphicRepFactory();
	QUuid dataID() const override;
	QString name() const override{return m_name;}
	QString mediaPath() const { return m_mediaPath; }

	int width() { return m_width; }
	int height() { return m_height; }

private:
	unsigned int m_width;
	unsigned int m_height;

	QString m_name;
	QUuid m_uuid;

	Affine2DTransformation *m_ijToXY;

	QString m_mediaPath;
	Seismic3DAbstractDataset* m_originDataset;
	VideoLayerGraphicRepFactory * m_repFactory;
};

#endif /* QTCUDAIMAGEVIEWER_QGLTEXTURE_H_ */
