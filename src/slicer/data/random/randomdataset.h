#ifndef RANDOMDATASET_H
#define RANDOMDATASET_H

#include <QObject>
#include <QVector2D>
#include <QList>
#include <QColor>

#include "idata.h"
#include "randomview3d.h"


class RandomView3D;
class RandomTexDataset;
class RandomGraphicRepFactory;

class RandomDataset: public IData
{
Q_OBJECT
public:
RandomDataset(WorkingSetManager * workingSet,RandomView3D* random, const QString &name, QObject *parent =0);
	virtual ~RandomDataset();

	//IData
	virtual IGraphicRepFactory *graphicRepFactory();
	QUuid dataID() const override;
	QString name() const override{return m_name;}

	RandomView3D* getRandom3d();

	void addDataset(RandomTexDataset *dataset);
	void removeDataset(RandomTexDataset *dataset);


	QList<RandomTexDataset*> datasets();

signals:
	void datasetAdded(RandomTexDataset*);
	void datasetRemoved(RandomTexDataset*);

private:
	QString m_name;
	QUuid m_uuid;


	QList<RandomTexDataset*> m_datasets;

	RandomView3D* m_random= nullptr;

	RandomGraphicRepFactory* m_repFactory;

};

#endif
