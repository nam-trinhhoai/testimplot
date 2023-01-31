
#ifndef __QFILEUTILS__
#define __QFILEUTILS__


bool mkdirPathIfNotExist(QString path);

QString makeFilePathWithSuffix(QString source, QString suffix, QString ext = ".xt");

QString getDirectoryUp(QString dir0);

QString getTmpFilename(QString dirPath, QString prefix, QString ext);

bool fileRemove(QString filename);


#endif
