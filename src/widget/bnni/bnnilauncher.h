#ifndef SRC_WIDGET_BNNI_BNNILAUNCHER_H_
#define SRC_WIDGET_BNNI_BNNILAUNCHER_H_

#include <QWidget>

class QToolButton;

class BnniLauncher : public QWidget {
	Q_OBJECT
public:
	BnniLauncher(QWidget* parent=nullptr, Qt::WindowFlags f = Qt::WindowFlags());
	~BnniLauncher();

public slots:
	void openInformation();
	void openTrainingSetCreator();
	void openLearningWidget();
	void openPredictionManager();

private:
	QToolButton* initToolButton(const QString& iconPath, const QString& text);
};

#endif
