#include "videoinformationpanelwidget.h"
#include "videoinformation.h"

#include <QFormLayout>
#include <QLabel>
#include <QStyle>
#include <QToolButton>
#include <QLineEdit>

#include <cstdlib>

VideoInformationPanelWidget::VideoInformationPanelWidget(VideoInformation* information, QWidget* parent) :
		IInformationPanelWidget(parent), m_information(information) {
	QString name = "No name";
	if (!m_information.isNull()) {
		name = m_information->name();
	}

	QFormLayout* mainLayout = new QFormLayout;
	setLayout(mainLayout);
	QLineEdit *labelName = new QLineEdit(name);
	labelName->setReadOnly(true);
	labelName->setStyleSheet("QLineEdit { border: none }");
	mainLayout->addRow("Name: ", labelName);
	QToolButton* playButton = new QToolButton;
	playButton->setIcon(style()->standardPixmap(QStyle::SP_MediaPlay));
	mainLayout->addRow("Play: ", playButton);

	connect(playButton, &QToolButton::clicked, this, &VideoInformationPanelWidget::play);
}

VideoInformationPanelWidget::~VideoInformationPanelWidget() {

}

void VideoInformationPanelWidget::play() {
	if (m_information.isNull()) {
		return;
	}

	QString path = m_information->mainPath();
	QString cmd = "vlc " + path;
	int returnVal = std::system(cmd.toStdString().c_str());
	if (returnVal!=0) {
		cmd = "totem " + path;
		std::system(cmd.toStdString().c_str());
	}
}

bool VideoInformationPanelWidget::saveChanges() {
	return false;
}
