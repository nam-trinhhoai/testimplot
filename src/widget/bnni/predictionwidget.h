#ifndef SRC_WIDGET_BNNI_PREDICTIONWIDGET_H_
#define SRC_WIDGET_BNNI_PREDICTIONWIDGET_H_

#include <QWidget>
#include <QProcess>

class QSpinBox;
class QLabel;
class QListWidget;
class ResourcesSelectorWidget;
class QLineEdit;

class PredictionWidget : public QWidget {
	Q_OBJECT
public:
	PredictionWidget(QWidget* parent=0, Qt::WindowFlags f = Qt::WindowFlags());
	~PredictionWidget();

	QRect volumeBounds() const;
	QRect generationBounds() const;

	void setProject(const QString& projectDirPath);

public slots:
	void run();
	void setYMin(int val);
	void setYMax(int val);
	void setZMin(int val);
	void setZMax(int val);
	void openProject();
	void generalizeSuffixChanged();

private slots:
	void resetTemporaryFile(QProcess::ProcessState state);

private:
	void updateVolumeBounds();
	void clearTrainingSetListWidget();
	void clearConfigListWidget();
	void clearCheckPointListWidget();
	void trainingSetChanged();
	void configChanged();
	void checkPointChanged();
	void loadSettings();

	QSpinBox* m_yMinSpinBox, *m_yMaxSpinBox, *m_zMinSpinBox, *m_zMaxSpinBox;
	QLineEdit* m_suffixLineEdit;
	ResourcesSelectorWidget* m_resourcesSelector;

	QLabel* m_projectLabel;
	QListWidget* m_trainingSetListWidget;
	QListWidget* m_configListWidget;
	QListWidget* m_checkPointListWidget;

	// rect.x=volume.y and rect.y=volume.z
	QRect m_volumeBounds;
	QRect m_generationBounds;
	int m_xmin;
	int m_xmax;

	QString m_projectDirPath;
	QString m_trainingSetDirPath;
	QString m_configFilePath;
	QString m_checkPointFilePath; // path without extension, remove .index extension
	QString m_generalizeSuffix = "_bnni";

	QProcess* m_process;
	QString m_processTemporaryFile = "";
};

#endif
