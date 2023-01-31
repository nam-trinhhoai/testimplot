/*
 * NextMainWindow.h
 *
 *  Created on: 22 juin 2018
 *      Author: l0380577
 */

#ifndef TARUMAPP_SRC_MAINWINDOW_NEXTMAINWINDOW_H_
#define TARUMAPP_SRC_MAINWINDOW_NEXTMAINWINDOW_H_

#include <QWidget>

#include "workingsetmanager.h"
#include "processrelay.h"

class GeotimeSystemInfo;

class QVBoxLayout;
class QMenuBar;
class QToolButton;

class TreeView;

class NextMainWindow : public QWidget{
	Q_OBJECT
public:
	NextMainWindow(QWidget* parent = nullptr);
	virtual ~NextMainWindow();

	QMenuBar* menuBar();

private:
	void initLaunchers();
	void geotimeLaunch();
	void spectrumVideoLaunch();
	void playVideoLaunch();
	void geotimeProcess();
	void openSystemInfo();
	QString getVersion();

	QToolButton* initToolButton(const QString& iconPath, const QString& text);

	//ProjectsRootItem root;
	QVBoxLayout* launchersLayout;
	QMenuBar* innerMenuBar;
	QToolButton* m_rgtButton;
	QToolButton* m_viewButton;
	QToolButton* m_conversionButton;
	QToolButton* m_dataBaseButton;
	QToolButton* m_videoButton;
	QToolButton* m_playVideoButton;
	QToolButton* m_bnniButton;
	QToolButton* m_ccusButton;
	QToolButton* m_petropyButton;
	QToolButton* m_systemInfoButton;
	QToolButton* m_calculatriceButton;
	QToolButton* m_screenShotButton;
	QToolButton* m_videoCaptureButton;
	QToolButton* m_vncButton;

	std::size_t m_idForWindows = 0;

	ProcessRelay* m_processRelay;

	GeotimeSystemInfo *m_systemInfo = nullptr;
};

#endif /* TARUMAPP_SRC_MAINWINDOW_NEXTMAINWINDOW_H_ */
