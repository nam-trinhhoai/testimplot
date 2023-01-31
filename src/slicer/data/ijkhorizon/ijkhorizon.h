#ifndef IJKHorizon_H
#define IJKHorizon_H

#include <QObject>
#include <memory>
#include "idata.h"
#include "iabstractisochrone.h"
#include "ifilebaseddata.h"

class Seismic3DAbstractDataset;

class MemoryIsochrone : public IAbstractIsochrone {
public:
	MemoryIsochrone(float* oriTab, long numTraces, long numProfils, bool takeOwnerShip);
	virtual ~MemoryIsochrone();

	virtual int getNumTraces() const override;
	virtual int getNumProfils() const override;

	virtual float getValue(long i, long j, bool* ok) override;
	virtual float* getTab() override;

private:
	long m_numTraces;
	long m_numProfils;
	float* m_buffer;
	bool m_ownBuffer;
	std::vector<float> m__vectorBuffer;
};

class IJKHorizon : public IData, public IFileBasedData {
Q_OBJECT
public:
	IJKHorizon(QString name, QString path, QString seismicOriginPath,
			WorkingSetManager *workingSet, QObject* parent=0);
	virtual ~IJKHorizon();

	//IData
	virtual IGraphicRepFactory* graphicRepFactory();
	QUuid dataID() const override;
	QString name() const override;

	QString path() const;
	QString seismicOriginPath() const;

	static bool filterHorizon(IJKHorizon* horizon, Seismic3DAbstractDataset* dataset);

	MemoryIsochrone* getIsochrone(); // caller take ownership of the returned object

private:
	QString m_name;
	QUuid m_uuid;

	QString m_path;
	QString m_seismicOriginPath;

	std::unique_ptr<IGraphicRepFactory> m_repFactory;
};

Q_DECLARE_METATYPE(IJKHorizon*)
#endif
