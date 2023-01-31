#ifndef SRC_PROCESS_KOHONENPROCESS_H_
#define SRC_PROCESS_KOHONENPROCESS_H_

#include <QString>

#include <vector>

class Seismic3DDataset;
class FixedLayerFromDataset;
class IAbstractIsochrone;


class AbstractKohonenProcess {
public:
	virtual ~AbstractKohonenProcess();

	// do computation with given arguments
	virtual bool compute(int exampleSize, int kohonenMapSize, int trainExamplesStep);

	static AbstractKohonenProcess* getObjectFromDataset(Seismic3DDataset* dataset, int channel);

	virtual bool setExtractionIsochrone(IAbstractIsochrone* isochrone);

	/**
	 * Set result holder
	 *
	 * If result holder is not defined, a new one will be created and the object will take ownership of it
	 * to retrieve it use getComputationResult (care about ownership) or releaseComputationResult (care about ownership)
	 */
	virtual bool setOutputHorizonProperties(FixedLayerFromDataset* saveProperties, QString tmapLabel);

	virtual bool setOutputHorizonBuffer(float* tab);
	virtual bool setOutputOnIsochroneAttribute(float* tab);

	// Return current result holder, it can be nullptr
	FixedLayerFromDataset* getComputationResult();
	QString tmapLabel() const;

	// Return currentresult holder and remove it from internal states and take ownership of it if the object had ownership
	FixedLayerFromDataset* releaseComputationResult();
	bool hasResultOwnership() const;

	bool isNonValue(float) const;

	virtual bool isExampleValid(long tabSize, long i, long j) const = 0;
	virtual bool getExample(float* tab, long tabSize, long i, long j) const = 0;

	long dimI() const;
	long dimJ() const;

protected:
	AbstractKohonenProcess(Seismic3DDataset* dataset, int channel);

	template<typename InputCubeType>
	struct CreateKohonenProcessObjectKernel {
		static AbstractKohonenProcess* run(Seismic3DDataset* dataset, int channel);
	};

	Seismic3DDataset* m_dataset = nullptr;
	int m_channel = 0;

	IAbstractIsochrone* m_isochrone = nullptr;
	FixedLayerFromDataset* m_computationResult = nullptr;
	float* m_outputHorizonBuffer = nullptr;
	float* m_outputOnIsochroneAttribute = nullptr;
	QString m_tmapLabel;
	bool m_haveResultOwnerShip = false;
	float m_nonValue = -9999.0;
	long m_dimI;
	long m_dimJ;
};

template<typename CubeType>
class KohonenProcess : public AbstractKohonenProcess {
	friend class AbstractKohonenProcess;
public:
	virtual ~KohonenProcess();

	virtual bool isExampleValid(long tabSize, long i, long j) const override;
	virtual bool getExample(float* tab, long tabSize, long i, long j) const override;

private:
	KohonenProcess(Seismic3DDataset* dataset, int channel);

	mutable std::vector<CubeType> m_buffer;
	long m_dimx;
	float m_tdeb;
	float m_pasech;
};

#include "KohonenProcess.hpp"

#endif
