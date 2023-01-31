#ifndef SRC_BNNI_DENSEGENETICPARAMS_H
#define SRC_BNNI_DENSEGENETICPARAMS_H

#include "structures.h"

#include <QObject>

class DenseGeneticParams : public QObject {
	Q_OBJECT
public:
	DenseGeneticParams(QObject* parent=nullptr);
	~DenseGeneticParams();

	QVector<unsigned int> layerSizes() const;
	void setLayerSizes(const QVector<unsigned int>& array);
	bool useDropout() const;
	void toggleDropout(bool val);
	double dropout() const;
	void setDropout(double val);
	bool useNormalisation() const;
	void toggleNormalisation(bool val);
	Activation activation() const;
	void setActivation(Activation val);

signals:
	void layerSizesChanged(QVector<unsigned int> array);
	void useDropoutChanged(bool val);
	void dropoutChanged(double val);
	void useNormalisationChanged(bool val);
	void activationChanged(Activation val);

private:
	QVector<unsigned int> m_layerSizes;
	bool m_useDropout = false;
	double m_dropout = 0.1;
	bool m_useNormalisation = false;
	Activation m_activation = Activation::selu;
};

#endif // SRC_BNNI_DENSEGENETICPARAMS_H
