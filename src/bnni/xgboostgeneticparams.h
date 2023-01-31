#ifndef SRC_BNNI_XGBOOSTGENETICPARAMS_H
#define SRC_BNNI_XGBOOSTGENETICPARAMS_H

#include <QObject>

class XgBoostGeneticParams : public QObject {
	Q_OBJECT
public:
	XgBoostGeneticParams(QObject* parent=nullptr);
	~XgBoostGeneticParams();

	int maxDepth() const;
	int nEstimators() const;
	double learningRate() const;
	double subsample() const;
	double colsampleByTree() const;

public slots:
	void setMaxDepth(int val);
	void setNEstimators(int val);
	void setLearningRate(double val);
	void setSubsample(double val);
	void setColsampleByTree(double val);

signals:
	void maxDepthChanged(int val);
	void nEstimatorsChanged(int val);
	void learningRateChanged(double val);
	void subsampleChanged(double val);
	void colsampleByTreeChanged(double val);

private:
	int m_maxDepth = 10;
	int m_nEstimators = 200;
	double m_learningRate = 0.1;
	double m_subsample = 0.1;
	double m_colsampleByTree = 0.2;
};

#endif // SRC_BNNI_XGBOOSTGENETICPARAMS_H
