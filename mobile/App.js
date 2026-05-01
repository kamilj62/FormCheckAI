import React, { useMemo, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

import * as ImagePicker from "expo-image-picker";
import * as DocumentPicker from "expo-document-picker";
import { Video, ResizeMode } from "expo-av";

const API_URL = "http://192.168.1.25:8000";

const formatLabel = (v) =>
  v
    ? String(v)
        .replaceAll("_", " ")
        .replace(/\b\w/g, (c) => c.toUpperCase())
    : "N/A";

const getStatusColor = (status) => {
  if (
    status === "poor" ||
    status === "incomplete" ||
    status === "drifting" ||
    status === "severe_flare"
  ) {
    return "#ef4444";
  }

  if (
    status === "borderline" ||
    status === "needs_work" ||
    status === "possible" ||
    status === "shallow" ||
    status === "fair" ||
    status === "limited_range" ||
    status === "possibly_shallow"
  ) {
    return "#f59e0b";
  }

  return "#22c55e";
};

const getScoreColor = (score) => {
  if (score >= 9) return "#22c55e";
  if (score >= 7.5) return "#84cc16";
  if (score >= 6) return "#f59e0b";
  return "#ef4444";
};

const getBestRep = (reps) => {
  if (!reps || reps.length === 0) return null;

  return reps.reduce((best, rep) => {
    return Number(rep.score || 0) > Number(best.score || 0) ? rep : best;
  }, reps[0]);
};

const getPhaseConfig = (exerciseLabel) => {
  const label = String(exerciseLabel || "").toLowerCase();

  if (label.includes("push press")) {
    return {
      title: "Push Press Phase Review",
      text: "Setup → Dip → Drive → Catch → Lockout",
      items: [
        ["setup", "Setup"],
        ["dip", "Dip"],
        ["drive", "Drive"],
        ["catch", "Catch"],
        ["lockout", "Lockout"],
      ],
    };
  }

  if (label.includes("squat")) {
    return {
      title: "Squat Phase Review",
      text: "Setup → Descent → Bottom → Ascent → Lockout",
      items: [
        ["setup", "Setup"],
        ["descent", "Descent"],
        ["bottom", "Bottom"],
        ["ascent", "Ascent"],
        ["lockout", "Lockout"],
      ],
    };
  }

  if (label.includes("bench")) {
    return {
      title: "Bench Press Phase Review",
      text: "Setup → Descent → Bottom → Press → Lockout",
      items: [
        ["setup", "Setup"],
        ["descent", "Descent"],
        ["bottom", "Bottom"],
        ["press", "Press"],
        ["lockout", "Lockout"],
      ],
    };
  }

  return {
    title: "Deadlift Phase Review",
    text: "Setup → Pull → Mid → Finish → Lockout",
    items: [
      ["setup", "Setup"],
      ["pull", "Pull"],
      ["mid", "Mid"],
      ["finish", "Finish"],
      ["lockout", "Lockout"],
    ],
  };
};

const getInteractiveZones = (result) => {
  const label = String(result?.exercise_label || "").toLowerCase();
  const reps = result?.rep_feedback || [];
  const bestRep = getBestRep(reps);
  const breakdown = bestRep?.breakdown || {};

  if (label.includes("push press")) {
    return [
      {
        id: "dip",
        title: "Dip",
        status: breakdown.dip || "good",
        note:
          breakdown.dip === "shallow"
            ? "Use a stronger vertical dip before driving overhead."
            : "Dip timing looks usable.",
      },
      {
        id: "knees",
        title: "Knees",
        status: breakdown.knees || "good",
        note:
          breakdown.knees === "borderline" || breakdown.knees === "poor"
            ? "Keep knees tracking over toes during the dip and drive."
            : "Knee tracking looks controlled.",
      },
      {
        id: "bar",
        title: "Bar Path",
        status: breakdown.bar_path || "good",
        note:
          breakdown.bar_path === "drifting"
            ? "Keep the bar close and drive straight overhead."
            : "Bar path looks controlled.",
      },
      {
        id: "lockout",
        title: "Lockout",
        status: breakdown.lockout || "good",
        note:
          breakdown.lockout === "incomplete"
            ? "Finish with elbows fully extended overhead."
            : "Overhead lockout looks solid.",
      },
    ];
  }

  if (label.includes("bench")) {
    return [
      {
        id: "wrists",
        title: "Wrists",
        status: breakdown.wrists || breakdown.lockout || "good",
        note: "Keep wrists stacked over elbows and avoid bending them back.",
      },
      {
        id: "elbows",
        title: "Elbows",
        status: breakdown.elbows || "good",
        note:
          breakdown.elbows === "poor" ||
          breakdown.elbows === "severe_flare" ||
          breakdown.elbows === "borderline"
            ? "Keep elbows controlled. Avoid aggressive flare."
            : "Elbow path looks controlled.",
      },
      {
        id: "bar",
        title: "Bar Path",
        status: breakdown.bar_path || breakdown.depth || "good",
        note:
          breakdown.depth === "limited_range" ||
          breakdown.depth === "possibly_shallow"
            ? "Use a full, controlled bar path from chest to lockout."
            : "Bar path looks controlled.",
      },
      {
        id: "lockout",
        title: "Lockout",
        status: breakdown.lockout || "good",
        note:
          breakdown.lockout === "incomplete"
            ? "Fully extend your arms at the top."
            : "Lockout looks solid.",
      },
    ];
  }

  if (label.includes("squat")) {
    return [
      {
        id: "torso",
        title: "Torso",
        status: breakdown.torso || "good",
        note:
          breakdown.torso === "poor" || breakdown.torso === "borderline"
            ? "Keep your chest taller and avoid folding forward."
            : "Torso angle looks controlled.",
      },
      {
        id: "depth",
        title: "Depth",
        status: breakdown.depth || "good",
        note:
          breakdown.depth === "poor" || breakdown.depth === "borderline"
            ? "Sink a little deeper while keeping your chest up."
            : "Depth looks good.",
      },
      {
        id: "knees",
        title: "Knees",
        status: breakdown.knees || "good",
        note:
          breakdown.knees === "poor" || breakdown.knees === "borderline"
            ? "Drive knees out and keep them tracking over toes."
            : "Knee tracking looks controlled.",
      },
      {
        id: "lockout",
        title: "Lockout",
        status: breakdown.lockout || "good",
        note: "Stand tall and finish each rep under control.",
      },
    ];
  }

  return [
    {
      id: "back",
      title: "Back Position",
      status: breakdown.back || "good",
      note:
        breakdown.back === "poor" || breakdown.back === "fair"
          ? "Brace your core and keep a neutral spine."
          : "Back position looks controlled.",
    },
    {
      id: "hips",
      title: "Hip Hinge",
      status: breakdown.hinge || "good",
      note:
        breakdown.hinge === "poor"
          ? "Push your hips back more before starting the pull."
          : "Hip hinge looks controlled.",
    },
    {
      id: "bar",
      title: "Bar Path",
      status: breakdown.bar_path || "good",
      note:
        breakdown.bar_path === "poor" || breakdown.bar_path === "drifting"
          ? "Keep the bar close to your body during the pull."
          : "Bar path looks controlled.",
    },
    {
      id: "lockout",
      title: "Lockout",
      status: breakdown.lockout || "good",
      note:
        breakdown.lockout === "incomplete" || breakdown.lockout === "poor"
          ? "Finish tall with hips and knees fully extended."
          : "Lockout looks solid.",
    },
  ];
};

const getZoneImagePath = (result, activeZone) => {
  const images = result?.phase_images || {};
  const label = String(result?.exercise_label || "").toLowerCase();
  const zoneId = activeZone?.id;

  if (label.includes("bench")) {
    if (zoneId === "wrists") return images.press || images.lockout || images.bottom;
    if (zoneId === "elbows") return images.press || images.bottom || images.lockout;
    if (zoneId === "bar") return images.descent || images.bottom || images.press;
    if (zoneId === "lockout") return images.lockout || images.press;
  }

  if (label.includes("push press")) {
    if (zoneId === "dip") return images.dip || images.setup;
    if (zoneId === "bar") return images.drive || images.catch;
    if (zoneId === "lockout") return images.lockout || images.catch;
  }

  if (label.includes("squat")) {
    if (zoneId === "depth") return images.bottom || images.descent;
    if (zoneId === "knees") return images.bottom || images.descent;
    if (zoneId === "torso") return images.descent || images.bottom;
    if (zoneId === "lockout") return images.lockout || images.ascent;
  }

  if (label.includes("deadlift")) {
    if (zoneId === "back") return images.pull || images.mid;
    if (zoneId === "bar") return images.mid || images.pull;
    if (zoneId === "lockout") return images.lockout || images.finish;
  }

  return (
    images.setup ||
    images.descent ||
    images.bottom ||
    images.press ||
    images.lockout ||
    images.pull ||
    images.mid ||
    images.finish ||
    null
  );
};

export default function App() {
  const [video, setVideo] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [visualsLoading, setVisualsLoading] = useState(false);
  const [selectedZone, setSelectedZone] = useState(null);

  const reset = () => {
    setResult(null);
    setSelectedZone(null);
    setVisualsLoading(false);
  };

  const recordWithCamera = async () => {
    reset();

    const permission = await ImagePicker.requestCameraPermissionsAsync();

    if (!permission.granted) {
      Alert.alert("Camera permission required");
      return;
    }

    const res = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
      videoMaxDuration: 60,
    });

    if (!res.canceled) {
      const a = res.assets[0];

      setVideo({
        uri: a.uri,
        name: a.fileName || "camera.mov",
        type: a.mimeType || "video/quicktime",
      });
    }
  };

  const pickFromLibrary = async () => {
    reset();

    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      Alert.alert("Library permission required");
      return;
    }

    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
    });

    if (!res.canceled) {
      const a = res.assets[0];

      setVideo({
        uri: a.uri,
        name: a.fileName || "library.mov",
        type: a.mimeType || "video/quicktime",
      });
    }
  };

  const pickFromCloud = async () => {
    reset();

    const res = await DocumentPicker.getDocumentAsync({
      type: "video/*",
      copyToCacheDirectory: true,
    });

    if (!res.canceled) {
      const a = res.assets[0];

      setVideo({
        uri: a.uri,
        name: a.name || "cloud.mov",
        type: a.mimeType || "video/quicktime",
      });
    }
  };

  const buildFormData = () => {
    const formData = new FormData();

    formData.append("file", {
      uri: video.uri,
      name: video.name,
      type: video.type,
    });

    return formData;
  };

  const generateVisuals = async () => {
    try {
      setVisualsLoading(true);

      const visualsRes = await fetch(`${API_URL}/generate_visuals`, {
        method: "POST",
        body: buildFormData(),
      });

      const visualsData = await visualsRes.json();
      console.log("VISUALS RESPONSE:", visualsData);

      if (!visualsRes.ok) {
        throw new Error(
          visualsData.detail ||
            visualsData.message ||
            "Visual generation request failed"
        );
      }

      setResult((prev) => ({
        ...prev,
        ...visualsData,
      }));
    } catch (err) {
      console.log("VISUALS ERROR:", err.message);

      setResult((prev) => ({
        ...prev,
        visuals_error: err.message,
      }));
    } finally {
      setVisualsLoading(false);
    }
  };

  const analyzeVideo = async () => {
    if (!video) {
      Alert.alert("Pick a video first");
      return;
    }

    setLoading(true);
    setResult(null);
    setSelectedZone(null);
    setVisualsLoading(false);

    try {
      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: buildFormData(),
      });

      const data = await res.json();
      console.log("ANALYZE RESPONSE:", data);

      if (!res.ok) {
        throw new Error(data.detail || data.message || "Analyze request failed");
      }

      setResult(data);
      setLoading(false);

      if (data?.rep_feedback?.length > 0) {
        await generateVisuals();
      }
    } catch (err) {
      setResult({ error: true, message: err.message });
      setLoading(false);
      setVisualsLoading(false);
    }
  };

  const reps = result?.rep_feedback || [];

  const avgScore = useMemo(() => {
    if (!reps.length) return null;

    const avg =
      reps.reduce((sum, rep) => sum + Number(rep.score || 0), 0) / reps.length;

    return Number(avg.toFixed(1));
  }, [reps]);

  const displayScore = avgScore !== null ? Math.round(avgScore * 10) : null;

  const bestRep = getBestRep(reps);

  const biggestFix =
    result?.set_summary?.biggest_fix ||
    bestRep?.feedback?.[0] ||
    "Upload a clear side-angle video for analysis.";

  const phaseConfig = getPhaseConfig(result?.exercise_label);
  const zones = getInteractiveZones(result);
  const activeZone = selectedZone || zones[0];

  const coachingImagePath = getZoneImagePath(result, activeZone);
  const coachingImageUrl = coachingImagePath
  ? `${API_URL}${coachingImagePath}?zone=${activeZone?.id || "default"}`
  : null;

  const overlayUrl = result?.overlay_video_url
    ? `${API_URL}${result.overlay_video_url}`
    : null;

  const phaseImages = result?.phase_images || {};

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.hero}>
          <View>
            <Text style={styles.eyebrow}>AI Movement Coach</Text>
            <Text style={styles.title}>FormCheck AI</Text>
            <Text style={styles.subtitle}>
              Upload a lift. Get rep scoring, coaching zones, phase images, and
              replay.
            </Text>
          </View>

          <View style={styles.logoBubble}>
            <Text style={styles.logoText}>AI</Text>
          </View>
        </View>

        <View style={styles.actionRow}>
          <TouchableOpacity
            style={styles.primaryButton}
            onPress={recordWithCamera}
          >
            <Text style={styles.primaryButtonText}>Record</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryButton}
            onPress={pickFromLibrary}
          >
            <Text style={styles.secondaryButtonText}>Library</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryButton}
            onPress={pickFromCloud}
          >
            <Text style={styles.secondaryButtonText}>Files</Text>
          </TouchableOpacity>
        </View>

        {video && (
          <View style={styles.selectedCard}>
            <Text style={styles.cardLabel}>Selected Video</Text>
            <Text style={styles.selectedName}>{video.name}</Text>
          </View>
        )}

        <TouchableOpacity
          style={[styles.analyzeButton, loading && styles.disabledButton]}
          onPress={analyzeVideo}
          disabled={loading}
        >
          {loading ? (
            <View style={styles.loadingBlock}>
              <ActivityIndicator color="#020617" />
              <Text style={styles.analyzeButtonText}>
                Analyzing movement...
              </Text>
            </View>
          ) : visualsLoading ? (
            <View style={styles.loadingBlock}>
              <ActivityIndicator color="#020617" />
              <Text style={styles.analyzeButtonText}>Building visuals...</Text>
            </View>
          ) : (
            <Text style={styles.analyzeButtonText}>Analyze Lift</Text>
          )}
        </TouchableOpacity>

        {result?.error && (
          <View style={styles.errorCard}>
            <Text style={styles.errorTitle}>Request Failed</Text>
            <Text style={styles.errorText}>{result.message}</Text>
          </View>
        )}

        {result?.feedback && !result.error && reps.length === 0 && (
          <View style={styles.errorCard}>
            <Text style={styles.errorTitle}>Video Not Usable</Text>
            <Text style={styles.errorText}>
              Camera angle is too close or unclear for reliable scoring.
            </Text>
            <Text style={styles.errorText}>
              Move farther back, record from the side, and keep the full body
              visible.
            </Text>
          </View>
        )}

        {result && !result.error && reps.length > 0 && (
          <>
            <View style={styles.dashboardGrid}>
              <View style={styles.scoreCard}>
                <Text style={styles.cardLabel}>Overall Score</Text>

                <View
                  style={[
                    styles.scoreCircle,
                    { borderColor: getScoreColor(avgScore || 0) },
                  ]}
                >
                  <Text style={styles.scoreBig}>{displayScore}</Text>
                  <Text style={styles.scoreSmall}>/100</Text>
                </View>

                <Text style={styles.exerciseName}>{result.exercise_label}</Text>
                <Text style={styles.confidenceText}>
                  Confidence {Math.round(Number(result.confidence || 0) * 100)}%
                </Text>
              </View>

              <View style={styles.insightCard}>
                <Text style={styles.cardLabel}>Biggest Fix</Text>
                <Text style={styles.bigFix}>{biggestFix}</Text>

                <View style={styles.miniStats}>
                  <View style={styles.statPill}>
                    <Text style={styles.statNumber}>
                      {result?.set_summary?.detected_reps || reps.length}
                    </Text>
                    <Text style={styles.statLabel}>Reps</Text>
                  </View>

                  <View style={styles.statPill}>
                    <Text style={styles.statNumber}>
                      {result?.set_summary?.best_rep || bestRep?.rep || "-"}
                    </Text>
                    <Text style={styles.statLabel}>Best</Text>
                  </View>

                  <View style={styles.statPill}>
                    <Text style={styles.statNumber}>
                      {result?.set_summary?.worst_rep || "-"}
                    </Text>
                    <Text style={styles.statLabel}>Needs Work</Text>
                  </View>
                </View>
              </View>
            </View>

            {visualsLoading && (
              <View style={styles.warningCard}>
                <ActivityIndicator color="#86efac" />
                <Text style={styles.warningTitle}>
                  Coaching visuals are loading
                </Text>
                <Text style={styles.warningText}>
                  Scores are ready. Phase images and replay will appear next.
                </Text>
              </View>
            )}

            {result?.visuals_error && (
              <View style={styles.errorCard}>
                <Text style={styles.errorTitle}>Visuals Could Not Load</Text>
                <Text style={styles.errorText}>{result.visuals_error}</Text>
              </View>
            )}

            <View style={styles.card}>
              <View style={styles.sectionHeader}>
                <View>
                  <Text style={styles.sectionTitle}>Rep Breakdown</Text>
                  <Text style={styles.sectionSub}>
                    Score trend across the set
                  </Text>
                </View>
              </View>

              {reps.map((rep) => {
                const score = Number(rep.score || 0);
                const barWidth = `${Math.min(100, Math.max(8, score * 10))}%`;

                return (
                  <View key={`rep-${rep.rep}`} style={styles.repRow}>
                    <View style={styles.repTop}>
                      <Text style={styles.repLabel}>Rep {rep.rep}</Text>
                      <Text style={styles.repScore}>{score.toFixed(1)}/10</Text>
                    </View>

                    <View style={styles.repBarTrack}>
                      <View
                        style={[
                          styles.repBarFill,
                          {
                            width: barWidth,
                            backgroundColor: getScoreColor(score),
                          },
                        ]}
                      />
                    </View>

                    <Text style={styles.repFeedback}>
                      {rep.feedback?.[0] || rep.issues?.[0] || "Good rep."}
                    </Text>
                  </View>
                );
              })}
            </View>

            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Interactive Coaching Map</Text>
              <Text style={styles.sectionSub}>
                Tap a zone to see the most relevant frame and coaching note.
              </Text>

              <View style={styles.coachImageWrap}>
                {coachingImageUrl ? (
                  <Image
                    key={`${activeZone?.id}-${coachingImagePath}`}
                    source={{ uri: coachingImageUrl }}
                    style={styles.coachImage}
                    resizeMode="contain"
                  />
                ) : (
                  <View style={styles.emptyImage}>
                    <Text style={styles.emptyImageText}>Visual loading...</Text>
                  </View>
                )}
              </View>

              <View style={styles.zoneGrid}>
                {zones.map((zone) => {
                  const isActive = activeZone?.id === zone.id;
                  const color = getStatusColor(zone.status);

                  return (
                    <TouchableOpacity
                      key={zone.id}
                      style={[
                        styles.zonePill,
                        isActive && styles.zonePillActive,
                        isActive && { borderColor: color },
                      ]}
                      onPress={() => setSelectedZone(zone)}
                    >
                      <View
                        style={[styles.statusDot, { backgroundColor: color }]}
                      />
                      <Text style={styles.zoneText}>{zone.title}</Text>
                    </TouchableOpacity>
                  );
                })}
              </View>

              {activeZone && (
                <View style={styles.coachingNote}>
                  <Text style={styles.noteTitle}>{activeZone.title}</Text>
                  <Text style={styles.noteStatus}>
                    Status: {formatLabel(activeZone.status)}
                  </Text>
                  <Text style={styles.noteText}>{activeZone.note}</Text>
                </View>
              )}
            </View>

            <View style={styles.card}>
              <Text style={styles.sectionTitle}>{phaseConfig.title}</Text>
              <Text style={styles.sectionSub}>{phaseConfig.text}</Text>

              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={styles.phaseScroller}
              >
                {phaseConfig.items.map(([key, label]) => {
                  const path = phaseImages[key];
                  const url = path ? `${API_URL}${path}` : null;

                  return (
                    <View key={key} style={styles.phaseCard}>
                      <View style={styles.phaseImageWrap}>
                        {url ? (
                          <Image
                            source={{ uri: url }}
                            style={styles.phaseImage}
                            resizeMode="cover"
                          />
                        ) : (
                          <View style={styles.emptyPhase}>
                            <Text style={styles.emptyPhaseText}>No image</Text>
                          </View>
                        )}
                      </View>
                      <Text style={styles.phaseLabel}>{label}</Text>
                    </View>
                  );
                })}
              </ScrollView>
            </View>

            {overlayUrl && (
              <View style={styles.card}>
                <Text style={styles.sectionTitle}>Coached Replay</Text>
                <Text style={styles.sectionSub}>
                  Overlay video with rep feedback and movement markers.
                </Text>

                <Video
                  source={{ uri: overlayUrl }}
                  style={styles.videoPlayer}
                  useNativeControls
                  resizeMode={ResizeMode.CONTAIN}
                />
              </View>
            )}

            <View style={styles.card}>
              <Text style={styles.sectionTitle}>Coach Summary</Text>

              <Text style={styles.summaryLine}>
                {result?.set_summary?.trend || "Form summary will appear here."}
              </Text>

              {result?.feedback?.map((item, index) => (
                <Text key={`feedback-${index}`} style={styles.feedbackLine}>
                  • {item}
                </Text>
              ))}
            </View>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#020617",
  },
  content: {
    padding: 18,
    paddingBottom: 42,
  },
  hero: {
    backgroundColor: "#0f172a",
    borderRadius: 28,
    padding: 22,
    borderWidth: 1,
    borderColor: "#1e293b",
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 16,
    marginBottom: 16,
  },
  eyebrow: {
    color: "#86efac",
    fontSize: 13,
    fontWeight: "800",
    letterSpacing: 1,
    textTransform: "uppercase",
    marginBottom: 8,
  },
  title: {
    color: "#f8fafc",
    fontSize: 34,
    fontWeight: "900",
    letterSpacing: -1,
  },
  subtitle: {
    color: "#94a3b8",
    fontSize: 15,
    lineHeight: 22,
    marginTop: 8,
    maxWidth: 280,
  },
  logoBubble: {
    width: 58,
    height: 58,
    borderRadius: 20,
    backgroundColor: "#86efac",
    alignItems: "center",
    justifyContent: "center",
  },
  logoText: {
    color: "#020617",
    fontSize: 20,
    fontWeight: "900",
  },
  actionRow: {
    flexDirection: "row",
    gap: 10,
    marginBottom: 14,
  },
  primaryButton: {
    flex: 1,
    backgroundColor: "#22c55e",
    borderRadius: 18,
    paddingVertical: 15,
    alignItems: "center",
  },
  primaryButtonText: {
    color: "#020617",
    fontWeight: "900",
    fontSize: 15,
  },
  secondaryButton: {
    flex: 1,
    backgroundColor: "#111827",
    borderRadius: 18,
    paddingVertical: 15,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#1f2937",
  },
  secondaryButtonText: {
    color: "#e5e7eb",
    fontWeight: "800",
    fontSize: 15,
  },
  selectedCard: {
    backgroundColor: "#0f172a",
    borderRadius: 22,
    padding: 16,
    borderWidth: 1,
    borderColor: "#1e293b",
    marginBottom: 14,
  },
  cardLabel: {
    color: "#64748b",
    fontSize: 12,
    fontWeight: "900",
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  selectedName: {
    color: "#f8fafc",
    fontSize: 15,
    fontWeight: "700",
    marginTop: 6,
  },
  analyzeButton: {
    backgroundColor: "#86efac",
    borderRadius: 22,
    paddingVertical: 18,
    alignItems: "center",
    marginBottom: 18,
  },
  disabledButton: {
    opacity: 0.75,
  },
  analyzeButtonText: {
    color: "#020617",
    fontSize: 17,
    fontWeight: "900",
  },
  loadingBlock: {
    alignItems: "center",
    gap: 8,
  },
  dashboardGrid: {
    gap: 14,
    marginBottom: 14,
  },
  scoreCard: {
    backgroundColor: "#0f172a",
    borderRadius: 28,
    padding: 22,
    borderWidth: 1,
    borderColor: "#1e293b",
    alignItems: "center",
  },
  scoreCircle: {
    width: 154,
    height: 154,
    borderRadius: 77,
    borderWidth: 12,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 18,
    marginBottom: 14,
    backgroundColor: "#020617",
  },
  scoreBig: {
    color: "#f8fafc",
    fontSize: 42,
    fontWeight: "900",
    lineHeight: 48,
  },
  scoreSmall: {
    color: "#94a3b8",
    fontWeight: "800",
  },
  exerciseName: {
    color: "#f8fafc",
    fontSize: 23,
    fontWeight: "900",
    marginTop: 4,
  },
  confidenceText: {
    color: "#94a3b8",
    marginTop: 6,
    fontWeight: "700",
  },
  insightCard: {
    backgroundColor: "#0f172a",
    borderRadius: 28,
    padding: 20,
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  bigFix: {
    color: "#f8fafc",
    fontSize: 22,
    fontWeight: "900",
    lineHeight: 29,
    marginTop: 10,
  },
  miniStats: {
    flexDirection: "row",
    gap: 10,
    marginTop: 18,
  },
  statPill: {
    flex: 1,
    backgroundColor: "#020617",
    borderRadius: 18,
    paddingVertical: 14,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  statNumber: {
    color: "#86efac",
    fontSize: 22,
    fontWeight: "900",
  },
  statLabel: {
    color: "#94a3b8",
    fontSize: 11,
    fontWeight: "800",
    marginTop: 3,
    textTransform: "uppercase",
  },
  card: {
    backgroundColor: "#0f172a",
    borderRadius: 28,
    padding: 18,
    borderWidth: 1,
    borderColor: "#1e293b",
    marginBottom: 14,
  },
  sectionHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  sectionTitle: {
    color: "#f8fafc",
    fontSize: 21,
    fontWeight: "900",
  },
  sectionSub: {
    color: "#94a3b8",
    fontSize: 13,
    fontWeight: "700",
    marginTop: 5,
    marginBottom: 14,
  },
  repRow: {
    marginTop: 14,
  },
  repTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 8,
  },
  repLabel: {
    color: "#e5e7eb",
    fontWeight: "900",
  },
  repScore: {
    color: "#f8fafc",
    fontWeight: "900",
  },
  repBarTrack: {
    height: 12,
    borderRadius: 999,
    backgroundColor: "#020617",
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  repBarFill: {
    height: "100%",
    borderRadius: 999,
  },
  repFeedback: {
    color: "#94a3b8",
    fontSize: 13,
    marginTop: 7,
    lineHeight: 18,
  },
  coachImageWrap: {
    height: 260,
    borderRadius: 22,
    overflow: "hidden",
    backgroundColor: "#020617",
    borderWidth: 1,
    borderColor: "#1e293b",
    marginBottom: 14,
  },
  coachImage: {
    width: "100%",
    height: "100%",
  },
  emptyImage: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  emptyImageText: {
    color: "#64748b",
    fontWeight: "800",
  },
  zoneGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  zonePill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 13,
    paddingVertical: 11,
    borderRadius: 999,
    backgroundColor: "#020617",
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  zonePillActive: {
    backgroundColor: "#111827",
    borderWidth: 2,
  },
  statusDot: {
    width: 9,
    height: 9,
    borderRadius: 999,
  },
  zoneText: {
    color: "#e5e7eb",
    fontWeight: "800",
    fontSize: 13,
  },
  coachingNote: {
    marginTop: 14,
    backgroundColor: "#020617",
    borderRadius: 20,
    padding: 16,
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  noteTitle: {
    color: "#f8fafc",
    fontSize: 18,
    fontWeight: "900",
  },
  noteStatus: {
    color: "#86efac",
    fontWeight: "900",
    marginTop: 6,
  },
  noteText: {
    color: "#cbd5e1",
    fontSize: 14,
    lineHeight: 21,
    marginTop: 8,
  },
  phaseScroller: {
    gap: 12,
    paddingRight: 12,
  },
  phaseCard: {
    width: 162,
    backgroundColor: "#020617",
    borderRadius: 20,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  phaseImageWrap: {
    height: 128,
    backgroundColor: "#111827",
  },
  phaseImage: {
    width: "100%",
    height: "100%",
  },
  emptyPhase: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  emptyPhaseText: {
    color: "#64748b",
    fontWeight: "800",
  },
  phaseLabel: {
    color: "#f8fafc",
    fontWeight: "900",
    padding: 12,
    textAlign: "center",
  },
  videoPlayer: {
    height: 260,
    borderRadius: 22,
    backgroundColor: "#020617",
  },
  warningCard: {
    backgroundColor: "#102018",
    borderRadius: 24,
    padding: 18,
    borderWidth: 1,
    borderColor: "#14532d",
    marginBottom: 14,
    gap: 8,
  },
  warningTitle: {
    color: "#bbf7d0",
    fontSize: 16,
    fontWeight: "900",
  },
  warningText: {
    color: "#dcfce7",
    lineHeight: 20,
  },
  errorCard: {
    backgroundColor: "#2a1111",
    borderRadius: 24,
    padding: 18,
    borderWidth: 1,
    borderColor: "#7f1d1d",
    marginBottom: 14,
  },
  errorTitle: {
    color: "#fecaca",
    fontSize: 17,
    fontWeight: "900",
    marginBottom: 8,
  },
  errorText: {
    color: "#fecaca",
    lineHeight: 21,
  },
  summaryLine: {
    color: "#e5e7eb",
    fontSize: 15,
    lineHeight: 22,
    marginTop: 10,
    marginBottom: 10,
  },
  feedbackLine: {
    color: "#94a3b8",
    lineHeight: 22,
    marginTop: 4,
  },
});