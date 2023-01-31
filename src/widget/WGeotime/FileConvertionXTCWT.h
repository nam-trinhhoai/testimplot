/*
 * 
 *
 *  Created on: 24 March 2020
 *      Author: l1000501
 */

#ifndef MURATAPP_SRC_TOOLS_XCOM_FILECONVERSIONXTCWT_H_
#define MURATAPP_SRC_TOOLS_XCOM_FILECONVERSIONXTCWT_H_

#include <QThread>
#include <QWidget>
#include <QString>
#include <QLineEdit>
#include <QLabel>
#include <QCheckBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QComboBox>
#include <QGroupBox>
#include <QDialog>
#include <QTableWidget>

#include "GeotimeProjectManagerWidget.h"

#include <vector>
#include <math.h>

class QTableView;
class QStandardItemModel;


class MyThread_file_convertion : public QThread
{
     // Q_OBJECT
	 public:
     MyThread_file_convertion();
     std::vector<QString> src_filenames;
     std::vector<QString> dst_filenames;
     std::vector<QString> tiny_filenames;
     int idxfile, idxfilemax, cpt, cpt_max, cont0, cont, complete, abort;
     QString current_filename;
	 // GeotimeConfigurationWidget *pp;
};

class MyThread_XTCWT_file_convertion : public MyThread_file_convertion
{
	public:
	MyThread_XTCWT_file_convertion();
	std::vector<float> cwt_error;
	protected:
	virtual void run() override;
};

class MyThread_FloatShort_file_convertion : public MyThread_file_convertion
{
	public:
	MyThread_FloatShort_file_convertion();
	protected:
	virtual void run() override;
};

class MyThread_CharShort_file_convertion : public MyThread_file_convertion
{
	public:
	MyThread_CharShort_file_convertion();
	protected:
	virtual void run() override;
};

class MyThread_Resample_file_convertion : public MyThread_file_convertion
{
	public:
	MyThread_Resample_file_convertion();
	protected:
	virtual void run() override;
};

class DialogFileConvertor : public QDialog
{
	Q_OBJECT
public:
	DialogFileConvertor(std::vector<QString> name, std::vector<QString> path, QWidget *parent=0);
};

class AbstractDialogConverter : public QWidget
{
	Q_OBJECT
public:
	AbstractDialogConverter(QWidget *parent=0);
protected:
	bool b_run;
	QPushButton *qpb_start, *qpb_abort, *qpb_exit;
	std::vector<QString> m_list;
	std::vector<QString> m_path;
	QProgressBar *qpb_progress;
	QTableWidget *tableWidget;
	QLabel *label_progress;
	MyThread_file_convertion* pthread;
	virtual void buttons_config(bool run);


protected slots:
	virtual void showTime();
	virtual void trt_start() = 0;
	virtual void trt_abort();
	virtual void trt_exit();
};

class DialogConvertionXTCWT : public AbstractDialogConverter
{
	Q_OBJECT
public:
	DialogConvertionXTCWT(std::vector<QString> name, std::vector<QString> path, QWidget *parent=0);
	// DialogConvertionXTCWT(QWidget *parent=0);
private:
	void run_conversion(std::vector<QString> src_filenames, std::vector<QString> dst_filenames,
		std::vector<QString> tiny_filenames, std::vector<float> cwt_error);

protected slots:
	virtual void trt_start() override;
};

class DialogFloatToShort : public AbstractDialogConverter

{
	Q_OBJECT
public:
	DialogFloatToShort(std::vector<QString> name, std::vector<QString> path, QWidget *parent=0);

private:
	void run_conversion(std::vector<QString> src_filenames, std::vector<QString> dst_filenames,
		std::vector<QString> tiny_filenames);
protected slots:
	virtual void trt_start() override;
};

class DialogCharToShort : public AbstractDialogConverter

{
	Q_OBJECT
public:
	DialogCharToShort(std::vector<QString> name, std::vector<QString> path, QWidget *parent=0);

private:
	void run_conversion(std::vector<QString> src_filenames, std::vector<QString> dst_filenames,
		std::vector<QString> tiny_filenames);
protected slots:
	virtual void trt_start() override;
};

class DialogResample : public AbstractDialogConverter

{
	Q_OBJECT
public:
	DialogResample(std::vector<QString> name, std::vector<QString> path, QWidget *parent=0);

private:
	void run_conversion(std::vector<QString> src_filenames, std::vector<QString> dst_filenames,
		std::vector<QString> tiny_filenames);
protected slots:
	virtual void trt_start() override;
};


class FileConversionXTCWT : public QWidget
{
	Q_OBJECT
public:
    FileConversionXTCWT(QWidget *parent=0);
    virtual ~FileConversionXTCWT();

private:
    QProgressBar *qpb_progress;
    QLabel *label_progress, *label_cwterror;
    QLineEdit *lineedit_cwterror;
    QString src_filename0, dst_filename0;
    QLineEdit *lineedit_srcfilename, *lineedit_dstfilename;
    float cwt_error0;
    // void run_conversion(std::vector<QString> src_filenames, std::vector<QString> dst_filenames, std::vector<QString> tiny_filenames, float cwt_error);

    QPushButton *qpb_ok, *qpb_cancel;
    QString src_name, src_fullname, src_ext, dst_ext;
    QLabel *label_dsterror, *label_title;
   	GeotimeProjectManagerWidget *projectmanager;

private slots:
	void trt_start0();
    void trt_session_load();
    void trt_session_save();
};


#endif
