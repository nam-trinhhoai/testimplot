#ifndef SRC_BNNI_MODELS_TREEPARAMETERSMODEL
#define SRC_BNNI_MODELS_TREEPARAMETERSMODEL

#include "baseparametersmodel.h"

class TreeParametersModel : public BaseParametersModel {
	Q_OBJECT
public:
	TreeParametersModel(QObject* parent=nullptr);
	~TreeParametersModel();

	unsigned int getNEstimator() const;
	void setNEstimator(unsigned int val);
	unsigned int getMaxDepth() const;
	void setMaxDepth(unsigned int val);
	double getSubSample() const;
	void setSubSample(double val);
	double getColSampleByTree() const;
	void setColSampleByTree(double val);

	bool loadNEstimator(QString txt);
	bool loadMaxDepth(QString txt);
	bool loadSubSample(QString txt);
	bool loadColSampleByTree(QString txt);

	virtual QString getExtension() const override;
    virtual bool validateArguments() override;
    virtual QStringList warningForArguments() override;

signals:
	void nEstimatorChanged(unsigned int val);
	bool maxDepthChanged(unsigned int val);
	void subSampleChanged(double val);
	void colSampleByTreeChanged(double val);

private:
	unsigned int m_nEstimator = 200;
	unsigned int m_maxDepth = 10;
	double m_subSample = 0.1;
	double m_colSampleByTree = 0.2;
};

#endif // SRC_BNNI_MODELS_TREEPARAMETERSMODEL
