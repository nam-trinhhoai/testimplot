#include "bnnilauncher.h"
#include "trainingsetmanagerwrapper.h"
#include "trainingsetparameterwidget.h"
#include "bnnimainwindow.h"
#include "predictionwidget.h"

#include <boost/filesystem/path.hpp>

#include <QToolButton>
#include <QHBoxLayout>
#include <QPixmap>
#include <QApplication>

namespace fs = boost::filesystem;

BnniLauncher::BnniLauncher(QWidget* parent, Qt::WindowFlags f) {
	setAttribute(Qt::WA_DeleteOnClose);
	setWindowTitle("BNNI Menu");

	QHBoxLayout* mainLayout = new QHBoxLayout;
	setLayout(mainLayout);

	this->setStyleSheet("BnniLauncher {background-color: rgba(202, 202, 202, 100%)}");

	QToolButton* informationButton = initToolButton(":/slicer/icons/Info.svg", "Information");
	mainLayout->addWidget(informationButton, 1);
	QToolButton* trainingSetButton = initToolButton(":/slicer/icons/mainwindow/BnniTrainingSet.svg", "Data TrainingSet");
	mainLayout->addWidget(trainingSetButton, 1);
	QToolButton* learningButton = initToolButton(":/slicer/icons/mainwindow/BnniLearning.svg", "Learning");
	mainLayout->addWidget(learningButton, 1);
	QToolButton* predictionButton = initToolButton(":/slicer/icons/mainwindow/BnniGeneralization.svg", "Generalization");
	mainLayout->addWidget(predictionButton, 1);

	connect(informationButton, &QToolButton::clicked, this, &BnniLauncher::openInformation);
	connect(trainingSetButton, &QToolButton::clicked, this, &BnniLauncher::openTrainingSetCreator);
	connect(learningButton, &QToolButton::clicked, this, &BnniLauncher::openLearningWidget);
	connect(predictionButton, &QToolButton::clicked, this, &BnniLauncher::openPredictionManager);

	this->setMinimumSize(350, 110);
	this->setMaximumSize(350, 110);
}

BnniLauncher::~BnniLauncher() {

}

void BnniLauncher::openInformation() {
	TrainingSetManagerWrapper* widget = new TrainingSetManagerWrapper;
	widget->setVisible(true);
}

void BnniLauncher::openTrainingSetCreator() {
	TrainingSetParameterWidget* widget = new TrainingSetParameterWidget;
	widget->setVisible(true);
}

void BnniLauncher::openLearningWidget() {
	BnniMainWindow* mainWindow = new BnniMainWindow;
	fs::path full_path(QApplication::applicationFilePath().toStdString());
	QString rootScriptDir = QString::fromStdString(full_path.parent_path().parent_path().string() + "/scripts");
	mainWindow->setProgramLocation(rootScriptDir + "/BNNI/");
	mainWindow->setInterfaceProgramLocation(rootScriptDir+"/BNNI_Interface/");
	mainWindow->setVisible(true);
}

void BnniLauncher::openPredictionManager() {
	PredictionWidget* widget = new PredictionWidget;
	widget->setVisible(true);
}

QToolButton* BnniLauncher::initToolButton(const QString& iconPath, const QString& text) {
	QToolButton* toolButton = new QToolButton;
	toolButton->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
	toolButton->setText(text);
	QFont font("Roboto");
	font.setPointSize(7);
	toolButton->setFont(font);
	QPixmap rgtPixmap(iconPath);
	QIcon rgtIcon(rgtPixmap);
	toolButton->setIcon(rgtIcon);
	toolButton->setIconSize(QSize(60, 60));
	toolButton->setStyleSheet("QToolButton {border-image: none; background-color: rgba(150, 150, 150, 0%); color: black; margin: 0px; padding: 0px}");
	toolButton->setMaximumWidth(300);
	toolButton->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	return toolButton;
}
