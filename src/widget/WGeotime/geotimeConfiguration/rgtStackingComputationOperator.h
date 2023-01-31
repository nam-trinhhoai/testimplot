
#ifndef __RGTVOLUMICVOLUMECOMPUTATIONOPERATOR__
#define __RGTVOLUMICVOLUMECOMPUTATIONOPERATOR__

#include <string>
#include <QPolygon>

#include <ivolumecomputationoperator.h>

class RgtStackingComputationOperator : public QObject, public IVolumeComputationOperator
{
public:
	RgtStackingComputationOperator();
	RgtStackingComputationOperator(std::string seismicFilename, std::string rgtFilename, std::string rgtName);
	void setSurveyPath(QString path);

	// ToTest(Seismic3DDataset* dataset, QObject* parent=0) : QObject(parent) { m_data = dataset; }

	virtual QString name();
	virtual QString getSurveyPath();
	virtual void computeInline(int z, void** buffer);
	virtual void computeXline(int y, void** buffer);
	virtual void computeRandom(QPolygon traces, void** buffer);
	virtual long dimI();
	virtual long dimJ();
	virtual long dimK();
	virtual long dimV();

	virtual double originI();
	virtual double originJ();
	virtual double originK();
	virtual double stepI();
	virtual double stepJ();
	virtual double stepK();
	virtual ImageFormats::QSampleType sampleType();
	virtual SampleUnit sampleUnit();
	void setData(float *data, int *size);
	void setDip(short *dipxy, short *dipxz);
	void setMask2D(char *mask2D);

	void setRgtStacking(void *p) { m_rgtStacking = p; };

private:
	std::string m_seismicFilename = "";
	std::string m_rgtFilename = "";
	std::string m_rgtName = "";
	long m_dimx = 1;
	long m_dimy = 1;
	long m_dimz = 1;
	double m_startSlice = 0.0;
	double m_startRecord = 0.0;
	double m_startSample = 0.0;
	double m_stepSlices = 1.0;
	double m_stepRecords = 1.0;
	double m_stepSamples = 1.0;
	float *m_tau = nullptr;
	int *m_scaleSize = nullptr;
	short *m_dipxy = nullptr;
	short *m_dipxz = nullptr;
	char *m_mask2D = nullptr;
	SampleUnit m_sampleUnit;
	QString m_surveyPath = "";
	// void dataScale(float* data, char *mask2D_, short *dipxy, short *dipxz, int *size, double vmin, double vmax);
	void dataScale(short* data, int *size, double vmin, double vmax);
	void integrateInplace(short *stack, long dimx, long dimy, long dimz);
	void transposeInplace(short *data, long dimx, long dimy, long dimz);
	void *m_rgtStacking = nullptr;
};


#endif
