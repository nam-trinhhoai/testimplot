#ifndef SRC_DATAMANAGER_UTIL_H_
#define SRC_DATAMANAGER_UTIL_H_

#include "deleteableleaf.h"

#include <QString>
#include <QStringList>
#include <QPair>

QPair<bool, QStringList> mkpath(QString path);
bool rmpath(QStringList paths);
bool stringListMatch(const QStringList& smallList, const QStringList& longList, long startIndex);
QString moveFileBetweenDirSystems(QString oriPath, const QStringList& _oriDirSystem, const QStringList& _targetDirSystem);
QString moveFileToTrash(QString oriPath);
QString restoreFileFromTrash(QString oriPath);
std::pair<bool, DeletableLeaf> ijkMoveLeaf(DeletableLeaf leaf, bool toTrash=true);
std::pair<bool, DeletableLeaf> ijkMoveToTrash(DeletableLeaf leaf);
std::pair<bool, DeletableLeaf> ijkRestoreFromTrash(DeletableLeaf leaf);
std::pair<bool, DeletableLeaf> ijkDeleteFromTrash(DeletableLeaf leaf, QString logPath);
bool deleteFile(QString path, QString logPath);
bool deleteDir(QString path, QString logPath);
bool deletePath(QString path, QString logPath);
void addToLog(QString logPath, QString logText);

#endif
