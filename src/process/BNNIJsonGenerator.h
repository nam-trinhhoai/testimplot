#ifndef SRC_PROCESS_BNNIJSONGENERATOR_H_
#define SRC_PROCESS_BNNIJSONGENERATOR_H_

#include "affine2dtransformation.h"
#include "affinetransformation.h"
#include "viewutils.h"
#include "wellbore.h"
#include "bnnitrainingset.h"

#include <QString>
#include <QList>
#include <memory>
#include <utility>

#include <nlohmann/json.hpp>

#include <rapidjson/document.h>

typedef rapidjson::GenericDocument<rapidjson::ASCII<>> WDocument;
typedef rapidjson::GenericValue<rapidjson::ASCII<> > WValue;
typedef rapidjson::GenericMember<rapidjson::ASCII<>, rapidjson::MemoryPoolAllocator<> > WMember;

class WellHead;
class MtLengthUnit;

class BnniJsonGenerator {
public:
	typedef struct DefinedDouble {
		double val = -9999.0;
		bool isDefined = false;
	} DefinedDouble;

	typedef struct WellSeismicVoxel {
		int i;
		int j;
		int k;
		std::vector<double> mds;
		std::vector<DefinedDouble> logs;
	} WellSeismicVoxel;

	typedef struct IJKPoint {
		int i;
		int j;
		int k;
	} IJKPoint;

	typedef struct IJKPointDouble {
		double i;
		double j;
		double k;
	} IJKPointDouble;

	class BnniWell {
	public:
		virtual ~BnniWell();
		virtual QString getUniqueName() const = 0;
		virtual int getLogsCount() const = 0;
		virtual QString getLogName(int logIdx) const = 0;
		virtual QString getLogKind(int logIdx) const = 0;
		virtual QString getLogUniqueName(int logIdx) const = 0;
		virtual QString getWellBoreSampleKey() const = 0;
	};

	class OriginWell : public BnniWell {
	public:
		OriginWell(const WellBore* wellBore);
		~OriginWell();
		virtual QString getUniqueName() const override;
		virtual int getLogsCount() const override;
		virtual QString getLogName(int logIdx) const override;
		virtual QString getLogKind(int logIdx) const override;
		virtual QString getLogUniqueName(int logIdx) const override;
		virtual QString getWellBoreSampleKey() const override;
	private:
		const WellBore* m_wellBore;
	};

	class AugmentedWell : public BnniWell {
	public:
		AugmentedWell();
		~AugmentedWell();
		virtual QString getUniqueName() const override;
		virtual int getLogsCount() const override;
		virtual QString getLogName(int logIdx) const override;
		virtual QString getLogKind(int logIdx) const override;
		virtual QString getLogUniqueName(int logIdx) const override;
		virtual QString getWellBoreSampleKey() const override;

		QString getAugmentationCode() const;

		void setOriginWellName(const QString& name);
		void setOriginWellDescPath(const QString& path);
		void setOriginWellLogNames(const std::vector<QString>& logNames);
		void setOriginWellLogKinds(const std::vector<QString>& logKinds);
		void setOriginWellLogPaths(const std::vector<QString>& logPaths);
		void setDy(int val);
		void setDz(int val);

	private:
		QString m_originWellName;
		QString m_originWellDescPath;
		std::vector<QString> m_originWellLogNames;
		std::vector<QString> m_originWellLogKinds;
		std::vector<QString> m_originWellLogPaths;
		int m_dy;
		int m_dz;
	};

	class WellModifierOperator {
	public:
		virtual ~WellModifierOperator();
		virtual bool isValid() const = 0;
		virtual IJKPointDouble convert(IJKPointDouble pt, bool& ok) const = 0;
	};

	class ShiftStretchSqueezeOperator : public WellModifierOperator {
	public:
		// dj, dk in voxel
		// i data in depth index
		ShiftStretchSqueezeOperator(double dj, double dk, const std::vector<double>& inputXData, const std::vector<double>& outXData);
		virtual ~ShiftStretchSqueezeOperator();

		bool isValid() const override;
		bool isInGslBounds(double val) const;
		IJKPointDouble convert(IJKPointDouble pt, bool& ok) const override;
	private:
		bool m_valid = false;
		double m_inputDataMin = 0;
		double m_inputDataMax = 1;
		double m_outputValueForMin = 0;
		double m_outputValueForMax = 0;

		bool m_useGsl = false;
		double m_a = 0;
		double m_b = 0;
		gsl_interp_accel* m_acc = nullptr;
		gsl_spline* m_spline_steffen = nullptr;
		double m_dj = 0;
		double m_dk = 0;
	};

	class MeanWellModifierOperator : public WellModifierOperator {
	public:
		MeanWellModifierOperator(const std::vector<std::shared_ptr<WellModifierOperator>>& ops);
		virtual ~MeanWellModifierOperator();

		virtual bool isValid() const override;
		virtual IJKPointDouble convert(IJKPointDouble pt, bool& ok) const override;
	private:
		std::vector<std::shared_ptr<WellModifierOperator>> m_ops;
	};

	class WellModifierOperatorGenerator {
	public:
		static std::shared_ptr<WellModifierOperatorGenerator> getGenerator(const std::vector<QString>& seismics, int augmentationDistance,
				double pasSampleSurrechantillon);
		~WellModifierOperatorGenerator();

		std::shared_ptr<WellModifierOperator> getOperator(int baseJ, int baseK, int offsetJ, int offsetK);

	private:
		WellModifierOperatorGenerator(const std::vector<QString>& seismics, int dimJ, int augmentationDistance, double pasSampleSurrechantillon);

		std::vector<QString> m_seismics;
		int m_augmentationDistance;
		long m_dimJ = 0;
		double m_pasSampleSurrechantillon;

		std::map<long, std::array<std::shared_ptr<WellModifierOperator>, 9>> m_cachedOperators;
	};

	class WellShiftedGenerator {
	public:
		WellShiftedGenerator(std::shared_ptr<WellModifierOperatorGenerator> opGenerator, int offsetJ, int offsetK);
		~WellShiftedGenerator();

		std::shared_ptr<WellModifierOperator> getOperator(int baseJ, int baseK);

	private:
		std::shared_ptr<WellModifierOperatorGenerator> m_opGenerator;
		int m_offsetJ;
		int m_offsetK;
	};


	BnniJsonGenerator();
	~BnniJsonGenerator();

	// pair defined as (min, max)
	bool addInputVolume(const QString& path, const std::pair<float, float>& dynamic);
	bool addWellBore(const QString& descPath, const QString& deviationPath,
			const QString tfpPath, const QString& tfpName, const std::vector<QString>& logPaths,
			const std::vector<QString>& logNames, const QString& wellHeadDescPath);
	bool addHorizonsInterval(const QString& topPath, float topDelta, const QString& bottomPath, float bottomDelta);

	void defineLogsNames(const std::vector<BnniTrainingSet::BnniWellHeader>& headers);

	int halfWindow() const;
	void setHalfWindow(int val);

	QString outputJsonFile();
	void setOutputJsonFile(const QString& newPath);

	double pasSampleSurrechantillon() const;
	void setPasSampleSurrechantillon(double sampleRate);

	float mdSamplingRate() const;
	void setMdSamplingRate(float val);

	bool isActivatedBandPass() const;
	double bandPassFrequency() const;
	void deactivateBandPass();
	void activateBandPass(double freq);

	bool useAugmentation() const;
	void setUseAugmentation(bool val);
	int augmentationDistance() const;
	void setAugmentationDistance(int dist);
	float gaussianNoiseStd() const;
	void setGaussianNoiseStd(float val);
	bool useCnxAugmentation() const;
	void toggleCnxAugmentation(bool val);

	// valid only if a volume as been added
	SampleUnit seismicUnit() const;

	const MtLengthUnit* depthUnit() const;
	void setDepthUnit(const MtLengthUnit* depthUnit);

	std::pair<bool, QString> run();


	static QString getWellBoreUniqueName(const WellBore* wellBore);
	static QString getWellBoreUniqueName(const QString& wellHeadDirName, const QString& wellBoreDirName);
	static QString getWellBoreSampleKey(const WellBore* wellBore);
	static QString getWellBoreSampleKey(const QString& wellHeadDirName, const QString& wellBoreDirName);
	static QString getWellBoreLogUniqueName(const WellBore* wellBore, int logIndex);
	static QString getWellBoreLogUniqueName(const QString& wellHeadDirName, const QString& wellBoreDirName, const QString& logFileName);

private:
	void clearWellBores();

	void extractSingleWellLog(const Logs& log, int logIndex, std::list<WellSeismicVoxel>& extractionList, WellBore* wellBore,
			WellShiftedGenerator* opGenerator=nullptr);
	std::pair<bool, IJKPoint> isPointInBoundingBox(WellUnit unit, double logKey, WellBore* wellBore, WellShiftedGenerator* opGenerator=nullptr);
	std::pair<std::list<WellSeismicVoxel>::iterator, int> findVoxel(std::list<WellSeismicVoxel>& extractionList,
			std::list<WellSeismicVoxel>::iterator begin, const IJKPoint& voxel);

	void augmentData(std::vector<std::list<WellSeismicVoxel>>& voxels, std::vector<std::shared_ptr<BnniWell>>& wells);

	void createJSON(const std::vector<std::vector<std::vector<double>>>& seismicBuffers,
			const std::vector<std::list<WellSeismicVoxel>>& voxels,
			const std::vector<std::shared_ptr<BnniWell>>& allWells);

	void defineExtractionParameters(WDocument& document, nlohmann::json& newDoc);
	void defineSeismicParameters(WDocument& document, nlohmann::json& newDoc);
	void defineLogsParameters(WDocument& document, nlohmann::json& newDoc, const std::vector<std::shared_ptr<BnniWell>>& allWells);
	void defineSamples(WDocument& document, nlohmann::json& newDoc, const std::vector<std::vector<std::vector<double>>>& seismicBuffers,
			const std::vector<std::list<BnniJsonGenerator::WellSeismicVoxel>>& voxels,
			const std::vector<std::shared_ptr<BnniWell>>& allWells);
	bool getSeismicAndSurveyNames(const QString& datasetPath, QString& seismicName, QString& surveyName);

	// check if horizon is compatible, if the horizon is compatible read it and put it in buffer
	bool readHorizon(const QString& path, std::vector<float>& buffer);
	void applyDelta(std::vector<float>& buffer, float delta);

	std::vector<QString> m_paths;
	std::vector<std::pair<float, float>> m_seismicDynamics;
	std::vector<float> m_seismicWeights;
	std::vector<WellBore*> m_wellBores;
	std::vector<WellHead*> m_wellHeads;
	std::vector<BnniTrainingSet::BnniWellHeader> m_wellHeaders;
	std::list<std::pair<std::vector<float>, std::vector<float>>> m_horizonIntervals; // pair order is : top, bottom and topVal <= bottomVal
	std::vector<std::pair<float, float>> m_horizonDeltas; // pair order is : top, bottom
	std::vector<std::pair<QString, QString>> m_horizonNames; // pair order is : top, bottom
	int m_halfWindow;

	SampleUnit m_seismicUnit = SampleUnit::NONE;
	const MtLengthUnit* m_depthUnit = nullptr;

	std::unique_ptr<AffineTransformation> m_sampleTransform;
	std::unique_ptr<AffineTransformation> m_sampleTransformSurrechantillon;
	std::unique_ptr<Affine2DTransformation> m_inlineXlineTransfoForInline;
	std::unique_ptr<Affine2DTransformation> m_inlineXlineTransfoForXline;
	std::unique_ptr<Affine2DTransformation> m_ijToXYTransfo;

	long m_numSamples = 0;
	long m_numTraces = 0;
	long m_numProfils = 0;

	// for horizon matching
	float m_startTrace = 0.0f;
	float m_stepTraces = 1.0f;
	float m_startProfil = 0.0f;
	float m_stepProfils = 1.0f;

	long m_numSamplesSurrechantillon = 0;
	double m_pasSampleSurrechantillon = 0.5;

	QString m_outputJsonFile = "";

	bool m_useDerivative = false;
	bool m_useBandPassHighFrequency = false;
	float m_bandPassHighFrequency;
	float m_mdSamplingRate = 1.0;
	bool m_useAugmentation = false;
	int m_augmentationDistance = 11; // 2*ws+1 with ws being the default window size for genetic algorithm search
	float m_gaussianNoiseStd = 0.001; // use 0 mean
	bool m_useCnxAugmentation = true;

	float HORIZON_NULL_VALUE = -9999.0f;
};

#endif
