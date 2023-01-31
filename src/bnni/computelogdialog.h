#ifndef COMPUTELOGDIALIG_H
#define COMPUTELOGDIALIG_H

#include <QDialog>
#include <QStringList>

class QSpinBox;
class QLineEdit;
class QDoubleValidator;
class QComboBox;

class ComputeLogDialog : public QDialog
{
	Q_OBJECT
public:
	explicit ComputeLogDialog(QStringList log_list, QWidget *parent = 0);
	~ComputeLogDialog();

	float getFrequency();
	void setFrequency(float freq);

	int getWaveletSize();
	void setWaveletSize(int wavelet_size);

	QString getLogDt();
	QString getLogAttribut();

	QString getLogName();
	void setLogName(QString);

private:
	QStringList log_list;

	float frequency;
	int wavelet_size;

	QLineEdit* nameLineEdit = nullptr;

	QLineEdit* freqLineEdit = nullptr;
	QDoubleValidator* freqValidator = nullptr;
	QSpinBox* waveletSizeSpinBox = nullptr;

	QComboBox* logDtComboBox = nullptr;
	QComboBox* logAttributComboBox = nullptr;

};

#endif //COMPUTELOGDIALIG_H
