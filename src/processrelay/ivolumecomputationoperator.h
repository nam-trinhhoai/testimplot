#ifndef SRC_PROCESSRELAY_IVOLUMECOMPUTATIONOPERATOR_H
#define SRC_PROCESSRELAY_IVOLUMECOMPUTATIONOPERATOR_H

#include "icomputationoperator.h"
#include "imageformats.h"
#include "viewutils.h"

#include <QString>
#include <QPolygon>

class IVolumeComputationOperator : public IComputationOperator {
public:
	// may not be always true, if an issue arise, consider using transforms
	virtual QString getSurveyPath() = 0;
	// buffer is an array of slices
	// buffer is of size dimV and output buffer slice size
	// if dimV==0 only buffer[0] should be updated
	virtual void computeInline(int z, void** buffer) = 0;
	virtual void computeXline(int y, void** buffer) = 0;
	virtual void computeRandom(QPolygon traces, void** buffer) = 0;

	virtual long dimI() = 0; // X
	virtual long dimJ() = 0; // Y
	virtual long dimK() = 0; // Z
	virtual long dimV() = 0; // number of channels

	virtual double originI() = 0;
	virtual double originJ() = 0;
	virtual double originK() = 0;

	virtual double stepI() = 0;
	virtual double stepJ() = 0;
	virtual double stepK() = 0;

	virtual ImageFormats::QSampleType sampleType() = 0; // default output buffer type for computeInline
	virtual SampleUnit sampleUnit() = 0; // axis type
};




// TO TEST
#include "seismic3ddataset.h"
#include "seismicsurvey.h"
#include "sampletypebinder.h"

template<typename InputType>
struct ReadTestKernel {
	static void run(Seismic3DDataset* dataset, int z, void** _buffer) {
		InputType** buffer = static_cast<InputType**>(static_cast<void*>(_buffer));
		std::vector<InputType> bufferMono;
		bufferMono.resize(dataset->width()*dataset->height()*dataset->dimV());
		dataset->readInlineBlock<InputType>(bufferMono.data(), z, z+1, false);

		for (long i=0; i<dataset->dimV(); i++) {
			memcpy(buffer[i], bufferMono.data()+i*dataset->width()*dataset->height(), dataset->width()*dataset->height()*sizeof(InputType));
		}
	}
};

class ToTest : public QObject, public IVolumeComputationOperator {
public:
	ToTest(Seismic3DDataset* dataset, QObject* parent=0) : QObject(parent) {
		m_data = dataset;
	}

	virtual QString name() override {
		return "copy " + m_data->name();
	}
	virtual QString getSurveyPath() override {
		return m_data->survey()->idPath();
	}
	virtual void computeInline(int z, void** buffer) override {
		SampleTypeBinder binder(sampleType());
		binder.bind<ReadTestKernel>(m_data, z, buffer);
	}
	virtual void computeXline(int y, void** buffer) override {}
	virtual void computeRandom(QPolygon traces, void** buffer) override {}

	virtual long dimI()override {
		return m_data->height();
	}
	virtual long dimJ() override {
		return m_data->width();
	}
	virtual long dimK() override {
		return m_data->depth();
	}
	virtual long dimV() override {
		return m_data->dimV();
	}

	virtual double originI() {
		return m_data->cubeSeismicAddon().getFirstSample();
	}
	virtual double originJ() override {
		return m_data->cubeSeismicAddon().getFirstXline();
	}
	virtual double originK() override {
		return m_data->cubeSeismicAddon().getFirstInline();
	}

	virtual double stepI() override {
		return m_data->cubeSeismicAddon().getSampleStep();
	}
	virtual double stepJ() override {
		return m_data->cubeSeismicAddon().getXlineStep();
	}
	virtual double stepK() override {
		return m_data->cubeSeismicAddon().getInlineStep();
	}

	virtual ImageFormats::QSampleType sampleType() override {
		return m_data->sampleType();
	}
	virtual SampleUnit sampleUnit() override {
		return m_data->cubeSeismicAddon().getSampleUnit();
	}
private:
	Seismic3DDataset* m_data;
};

#endif // SRC_PROCESSRELAY_IVOLUMECOMPUTATIONOPERATOR_H
