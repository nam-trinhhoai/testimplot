#ifndef SRC_INFORMATIONMANAGER_VIDEOS_VIDEOINFORMATIONPANELWIDGET_H
#define SRC_INFORMATIONMANAGER_VIDEOS_VIDEOINFORMATIONPANELWIDGET_H

#include "iinformationpanelwidget.h"

#include <QColor>
#include <QPointer>

class VideoInformation;

class QPushButton;

class VideoInformationPanelWidget : public IInformationPanelWidget {
	Q_OBJECT
public:
	VideoInformationPanelWidget(VideoInformation* information, QWidget* parent=nullptr);
	virtual ~VideoInformationPanelWidget();

	virtual bool saveChanges() override;

public slots:
	void play();

private:
	QPointer<VideoInformation> m_information;
};

#endif // SRC_INFORMATIONMANAGER_VIDEOS_VIDEOINFORMATIONPANELWIDGET_H
