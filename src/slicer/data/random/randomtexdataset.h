#ifndef RANDOMTEXDATASET_H
#define RANDOMTEXDATASET_H

#include <QObject>
#include <QVector2D>
#include <QList>
#include <QColor>

#include "idata.h"


class RandomTexGraphicRepFactory;
class CudaImageTexture;
class RandomView3D;

class RandomTexDataset: public IData
{
Q_OBJECT
public:
RandomTexDataset(WorkingSetManager * workingSet, const QString &name, CudaImageTexture* texture,QVector2D range,RandomView3D *parent =0);
	virtual ~RandomTexDataset();

	//IData
	virtual IGraphicRepFactory *graphicRepFactory();
	QUuid dataID() const override;
	QString name() const override{return m_name;}

	CudaImageTexture* texture();
	QVector2D range();

	RandomView3D* parentRandom();

private:
	QString m_name;
	QUuid m_uuid;

	CudaImageTexture* m_texture = nullptr;
	QVector2D m_range;

	RandomView3D* m_randomParent= nullptr;




	RandomTexGraphicRepFactory* m_repFactory;

};

#endif
